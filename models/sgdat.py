# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容导入（允许作为包或脚本）
try:
    from .modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath
except Exception:
    from models.modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath


# -----------------------------
# Utils
# -----------------------------
def normalize_xyz(xyz, eps=1e-6):
    """
    xyz: (B, N, 3)
    以每个 batch 的中心和尺度进行归一化（保留相对几何结构）
    """
    center = xyz.mean(dim=1, keepdim=True)            # (B,1,3)
    xyz_centered = xyz - center
    scale = torch.sqrt((xyz_centered ** 2).sum(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]) + eps  # (B,1,1)
    xyz_normed = xyz_centered / scale
    return xyz_normed


def batched_index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, M)
    return: (B, M, C)
    """
    B, N, C = points.shape
    _, M = idx.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1).repeat(1, M)  # (B,M)
    return points[batch_indices, idx, :]


def squared_distance(src, dst):
    """
    src: (B, N, 3)
    dst: (B, M, 3)
    return: dist (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    xx = (src ** 2).sum(dim=-1, keepdim=True)         # (B,N,1)
    yy = (dst ** 2).sum(dim=-1).unsqueeze(1)          # (B,1,M)
    xy = torch.bmm(src, dst.transpose(1, 2))          # (B,N,M)
    dist = xx + yy - 2 * xy
    dist = torch.clamp(dist, min=0.0)
    return dist


def nearest_interpolate(target_xyz, source_xyz, source_feat):
    """
    使用最近邻把 source 的特征插值到 target
    target_xyz: (B, Nt, 3)
    source_xyz: (B, Ns, 3)
    source_feat:(B, Ns, C)
    return:     (B, Nt, C)
    """
    dist = squared_distance(target_xyz, source_xyz)          # (B,Nt,Ns)
    idx = dist.argmin(dim=-1)                                # (B,Nt)
    feat = batched_index_points(source_feat, idx)            # (B,Nt,C)
    return feat


@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    """
    纯 PyTorch CUDA 版 FPS（无需自定义 kernel）
    xyz: (B, N, 3)
    return: idx (B, npoint) Long
    """
    device = xyz.device
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist = ((xyz - centroid) ** 2).sum(-1)                   # (B,N)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)

    return idx


# -----------------------------
# Basic Blocks
# -----------------------------
class SharedMLP1D(nn.Module):
    """
    对 (B, N, Cin) 的特征做逐点卷积（Conv1d），输出 (B, N, Cout)
    新增：可选 dropout（默认 0，不改变原行为）
    """
    def __init__(self, cin, cout, bn=True, act=True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv1d(cin, cout, kernel_size=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(cout))
        if act:
            layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, C) -> (B,C,N) -> conv1d -> (B,N,Cout)
        x = x.permute(0, 2, 1).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


# -----------------------------
# SGDAT (轻量三层 U 形结构)
# -----------------------------
class SGDAT(nn.Module):
    def __init__(
        self,
        num_classes,
        base_dim=64,
        max_points=8000,
        debug=False,
        # ====== 抗过拟合：全部由外部传入，默认关闭 ======
        dropout_p: float = 0.0,        # MLP/上采样后的小概率 Dropout
        droppath_prob: float = 0.0,    # 分支级随机深度
        drop_rgb_p: float = 0.0,       # 训练时随机将 RGB 通道置零的概率
        drop_normal_p: float = 0.0,    # 训练时随机将 Normal 通道置零的概率
        logit_temp: float = 1.0,       # 温度缩放（1.0 表示关闭）
        use_channel_ccc: bool = False  # 可选通道注意力
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.max_points = max_points
        self.debug = debug

        # —— 不再读取 Config，全部来自构造参数 ——
        self.dropout_p = float(dropout_p)
        self.droppath_prob = float(droppath_prob)
        self.drop_rgb_p = float(drop_rgb_p)
        self.drop_normal_p = float(drop_normal_p)
        self.logit_temp = float(logit_temp)
        self.use_channel_ccc = bool(use_channel_ccc)

        # Encoder level-0 (N points)
        # 输入是 9 维：XYZ + RGB + Normal
        self.enc0 = SharedMLP1D(9, base_dim, dropout=self.dropout_p)  # -> (B,N,64)

        # 512 层的特征提炼：把 (XYZ_norm 3) + enc0_down(64) => 64
        self.enc512 = SharedMLP1D(3 + base_dim, base_dim, dropout=self.dropout_p)  # -> (B,512,64)

        # 128 层的特征提炼：把 (XYZ_norm 3) + enc512_down(64) => 128
        self.enc128 = SharedMLP1D(3 + base_dim, base_dim * 2, dropout=self.dropout_p)  # -> (B,128,128)

        # 可选通道注意力（不会改变尺寸）
        if self.use_channel_ccc:
            self.ccc_512 = ChannelCCC(base_dim)
            self.ccc_128 = ChannelCCC(base_dim * 2)
        else:
            self.ccc_512 = nn.Identity()
            self.ccc_128 = nn.Identity()

        # 位置编码（只用几何，避免把 RGB/Normal 混进位置）
        self.pos512 = SharedMLP1D(3, base_dim * 2, dropout=self.dropout_p)   # 3 -> 128
        self.posN   = SharedMLP1D(3, base_dim, dropout=self.dropout_p)       # 3 -> 64

        # Decoder 上采样整合
        # fuse_512 = cat([feat_512(64), up128_to_512(128), pos512(128)]) = 320
        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1),  # 64 + 128 + 128 = 320 -> 128
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        # fuse_N = cat([up512_to_N(128), enc0(64), posN(64)]) = 256
        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim * 2, 1),  # 128 + 64 + 64 = 256 -> 128
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        # DropPath 分支（对融合分支做结构化丢弃）
        self.branch_dp1 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()
        self.branch_dp2 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()

        # 分类头
        self.head = nn.Conv1d(base_dim * 2, num_classes, kernel_size=1)

    # --------- core ops ----------
    def _sample_and_gather(self, xyz, feat, npoint):
        """
        xyz:  (B,N,3)
        feat: (B,N,C)
        return:
          xyz_s:  (B,npoint,3)
          feat_s: (B,npoint,C)
          idx:    (B,npoint)
        """
        idx = farthest_point_sample(xyz, npoint)                  # (B,npoint)
        xyz_s = batched_index_points(xyz, idx)                    # (B,npoint,3)
        feat_s = batched_index_points(feat, idx)                  # (B,npoint,C)
        return xyz_s, feat_s, idx

    def _feature_group_dropout(self, points):
        """
        对输入 (B,N,9) 在训练时进行分组随机丢弃：
        - 以 drop_rgb_p 概率将 RGB (3:6) 置 0
        - 以 drop_normal_p 概率将 Normal (6:9) 置 0
        """
        if not self.training:
            return points
        out = points
        if self.drop_rgb_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_rgb_p:
            out = out.clone()
            out[:, :, 3:6] = 0.0
        if self.drop_normal_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_normal_p:
            out = out.clone()
            out[:, :, 6:9] = 0.0
        return out

    def forward(self, points):
        """
        points: (B, N, 9) -> [xyz(0:3), rgb(3:6), normal(6:9)]
        return: logits (B, N, num_classes)
        """
        B, N, C = points.shape
        assert C == 9, f"Expect input features 9 (xyz+rgb+normal), but got {C}"

        if self.debug:
            print(f"[DEBUG] Input shape: {points.shape}")

        # 训练时可选特征分组丢弃（只影响 RGB/Normal，几何保持）
        points = self._feature_group_dropout(points)

        # 拆分
        xyz = points[:, :, 0:3]  # (B,N,3)

        # 仅对 xyz 做归一化用于几何处理
        xyz_normed = normalize_xyz(xyz)                          # (B,N,3)

        # Encoder-0: 直接把 9 维输入嵌入到 64
        feat0 = self.enc0(points)                                # (B,N,64)

        # 下采样到 512
        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0, npoint=512)  # (B,512,3),(B,512,64)
        # 在 512 处再堆一个编码： [xyz_512(3) + feat0_512(64)] -> 64
        enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)     # (B,512,67)
        feat_512 = self.enc512(enc_512_in)                       # (B,512,64)
        feat_512 = self.ccc_512(feat_512)                        # (B,512,64)

        # 再下采样到 128
        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)  # (B,128,3),(B,128,64)
        # 在 128 处编码： [xyz_128(3) + feat_512_128(64)] -> 128
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)  # (B,128,67)
        feat_128 = self.enc128(enc_128_in)                       # (B,128,128)
        feat_128 = self.ccc_128(feat_128)                        # (B,128,128)

        if self.debug:
            with torch.no_grad():
                n_nan1 = torch.isnan(feat_512).any().item()
                n_nan2 = torch.isnan(feat_128).any().item()
                print(f"[DEBUG] out1 shape: {feat_512.shape}, NaN={int(n_nan1)}, Inf={int(torch.isinf(feat_512).any().item())}")
                print(f"[DEBUG] out2 shape: {feat_128.shape}, NaN={int(n_nan2)}, Inf={int(torch.isinf(feat_128).any().item())}")

        # Decoder: 128 -> 512
        up128_to_512 = nearest_interpolate(xyz_512, xyz_128, feat_128)  # (B,512,128)

        # 512 的位置编码（仅 xyz）
        pos512 = self.pos512(xyz_512)                             # (B,512,128)

        # 分支级 DropPath（结构化丢弃），仅在训练生效
        up128_to_512 = self.branch_dp1(up128_to_512)
        pos512 = self.branch_dp1(pos512)

        # 融合（确保 64 + 128 + 128 = 320）
        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)  # (B,512,320)
        # 走 up1
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,128)

        # Decoder: 512 -> N
        up512_to_N = nearest_interpolate(xyz_normed, xyz_512, fuse_512)  # (B,N,128)

        # N 的位置编码
        posN = self.posN(xyz_normed)                                    # (B,N,64)

        # 分支级 DropPath
        up512_to_N = self.branch_dp2(up512_to_N)
        posN = self.branch_dp2(posN)

        # 融合（确保 128 + 64 + 64 = 256）
        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)           # (B,N,256)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,128)

        # 分类头
        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,num_classes)

        # 温度缩放（可选）
        if self.logit_temp and self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        if self.debug:
            with torch.no_grad():
                print(f"[DEBUG] logits shape: {logits.shape}")

        return logits

    def get_loss(
        self,
        points,
        labels,
        class_weights=None,
        ignore_index=-1,
        aux_weight=None,
        reduction='mean',
        # ====== 新增损失选项（可选，默认关闭）======
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        **kwargs
    ):
        """
        points: (B, N, 9)
        labels: (B, N)
        """
        logits = self.forward(points)  # (B,N,K)
        B, N, K = logits.shape
        labels = labels.view(B, N)

        valid_mask = labels != ignore_index
        if valid_mask.sum() == 0:
            loss = logits.new_tensor(0.0, requires_grad=True)
            with torch.no_grad():
                acc = logits.new_tensor(0.0)
            stats = {
                "loss": float(loss.detach().item()),
                "acc": float(acc.detach().item()),
                "valid_points": int(valid_mask.sum().item())
            }
            return loss, logits, stats

        logits_valid = logits[valid_mask]     # (M,K)
        labels_valid = labels[valid_mask]     # (M,)

        weight = None
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weight = class_weights.to(logits.device, dtype=logits.dtype)
            else:
                weight = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)

        # ------ 统一用 log_probs & soft targets 的形式，兼容 label smoothing & focal ------
        log_probs = F.log_softmax(logits_valid, dim=-1)  # (M,K)
        probs = log_probs.exp()

        if label_smoothing is not None and label_smoothing > 0.0:
            # 构建平滑标签：(1-ε)one_hot + ε/K
            eps = float(label_smoothing)
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(eps / K)
            true_dist.scatter_(1, labels_valid.unsqueeze(1), 1.0 - eps)
        else:
            # 标准 one-hot（等价于交叉熵）
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, labels_valid.unsqueeze(1), 1.0)

        # 基础 CE（逐样本）
        ce = -(true_dist * log_probs).sum(dim=-1)  # (M,)

        # 类权重：按真实类的权重（即便有 smoothing 也常按主类计权）
        if weight is not None:
            w = weight[labels_valid]  # (M,)
            ce = ce * w

        # Focal：基于目标分布下的 pt
        if focal_gamma is not None and focal_gamma > 0.0:
            gamma = float(focal_gamma)
            pt = (true_dist * probs).sum(dim=-1).clamp(min=1e-6, max=1.0)  # (M,)
            focal_weight = (1.0 - pt) ** gamma
            ce = focal_weight * ce

        # 归约
        if reduction == 'mean':
            loss_main = ce.mean()
        elif reduction == 'sum':
            loss_main = ce.sum()
        else:
            loss_main = ce  # 'none'

        # 预留辅助损失接口（保持兼容）
        if aux_weight is not None and aux_weight != 0:
            loss = loss_main * (1 - aux_weight)  # 暂时简单处理
        else:
            loss = loss_main

        with torch.no_grad():
            pred = logits_valid.argmax(dim=-1)
            correct = (pred == labels_valid).sum()
            acc = correct.float() / labels_valid.numel()

        stats = {
            "loss": float(loss.detach().item()),
            "acc": float(acc.detach().item()),
            "valid_points": int(valid_mask.sum().item())
        }
        return loss, logits, stats
