# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容导入（允许作为包或脚本）
try:
    from .modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion
except Exception:
    from models.modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion


# -------------------------
# 数值更稳的工具函数
# -------------------------
def normalize_xyz(xyz, eps=1e-6):
    """
    对每个样本的点集做居中 + 统一尺度归一化，带极值保护。
    xyz: (B, N, 3)
    """
    center = xyz.mean(dim=1, keepdim=True)
    xyz_centered = xyz - center
    # 使用每批次内最大半径作尺度；加入 eps 防 0；再用 clamp 防 NaN/Inf
    # 半径 = 每个点的范数，取最大
    radius = torch.sqrt(torch.clamp((xyz_centered ** 2).sum(dim=-1, keepdim=True), min=0.0)).max(dim=1, keepdim=True)[0]
    scale = torch.clamp(radius, min=eps)
    xyz_normed = xyz_centered / scale
    return xyz_normed


def batched_index_points(points, idx):
    """
    points: (B, N, C), idx: (B, M)
    return: (B, M, C)
    """
    B, N, C = points.shape
    _, M = idx.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1).repeat(1, M)
    return points[batch_indices, idx, :]


def squared_distance(src, dst):
    """
    src: (B, N, C), dst: (B, M, C)  ->  (B, N, M)
    使用数值安全的计算；并下界裁剪到 0。
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    xx = (src ** 2).sum(dim=-1, keepdim=True)              # (B, N, 1)
    yy = (dst ** 2).sum(dim=-1).unsqueeze(1)               # (B, 1, M)
    xy = torch.bmm(src, dst.transpose(1, 2))               # (B, N, M)
    dist = xx + yy - 2 * xy
    # 极少数情况下可能出现 -1e-7 这类负值，裁剪为 0
    dist = torch.clamp(dist, min=0.0)
    return dist


def nearest_interpolate(target_xyz, source_xyz, source_feat):
    """
    对 target_xyz 中每个点，找到 source_xyz 最近邻并拷贝特征。
    target_xyz: (B, Nt, 3)
    source_xyz: (B, Ns, 3)
    source_feat: (B, Ns, C)
    return: (B, Nt, C)
    """
    dist = squared_distance(target_xyz, source_xyz)        # (B, Nt, Ns)
    idx = dist.argmin(dim=-1)                              # (B, Nt)
    feat = batched_index_points(source_feat, idx)          # (B, Nt, C)
    return feat


@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    """
    远点采样（无梯度以提升稳定与速度）
    xyz: (B, N, 3)
    return: idx (B, npoint)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)   # (B,1,3)
        dist = ((xyz - centroid) ** 2).sum(-1)                    # (B,N)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)
    return idx


# -------------------------
# 组件
# -------------------------
class SharedMLP1D(nn.Module):
    def __init__(self, cin, cout, bn=True, act=True, dropout: float = 0.0,
                 bn_eps: float = 1e-3, bn_momentum: float = 0.01):
        super().__init__()
        layers = [nn.Conv1d(cin, cout, kernel_size=1, bias=not bn)]
        if bn:
            bn1 = nn.BatchNorm1d(cout, eps=bn_eps, momentum=bn_momentum)
            layers.append(bn1)
        if act:
            layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 输入 (B,N,C) -> Conv1d 需要 (B,C,N)
        x = x.permute(0, 2, 1).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


# -------------------------
# 主干网络
# -------------------------
class SGDAT(nn.Module):
    def __init__(
        self,
        num_classes,
        base_dim=64,
        max_points=8000,
        debug=False,
        # ====== 抗过拟合 ======
        dropout_p: float = 0.0,
        droppath_prob: float = 0.0,
        drop_rgb_p: float = 0.0,
        drop_normal_p: float = 0.0,
        logit_temp: float = 1.0,
        # ====== 模块开关 ======
        use_channel_ccc: bool = False,
        use_dynamic_fusion: bool = True,
        use_linear_gva: bool = True,
        # 动态邻域参数
        dyn_neighbors: int = 16,
        dyn_min_radius: float = 0.02,
        dyn_max_radius: float = 0.30,
        # ====== BN 稳定性参数 ======
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.max_points = max_points
        self.debug = debug

        self.dropout_p = float(dropout_p)
        self.droppath_prob = float(droppath_prob)
        self.drop_rgb_p = float(drop_rgb_p)
        self.drop_normal_p = float(drop_normal_p)
        self.logit_temp = float(logit_temp)

        self.use_channel_ccc = bool(use_channel_ccc)
        self.use_dynamic_fusion = bool(use_dynamic_fusion)
        self.use_linear_gva = bool(use_linear_gva)

        self.dyn_neighbors = dyn_neighbors
        self.dyn_min_radius = dyn_min_radius
        self.dyn_max_radius = dyn_max_radius

        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)

        # Encoder level-0 (N points): 输入 9 维
        self.enc0 = SharedMLP1D(9, base_dim, dropout=self.dropout_p,
                                bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)  # (B,N,64)

        # 512 层编码： [xyz(3)+feat64] -> 64
        self.enc512 = SharedMLP1D(3 + base_dim, base_dim, dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)  # (B,512,64)

        # 128 层编码： [xyz(3)+feat64] -> 128
        self.enc128 = SharedMLP1D(3 + base_dim, base_dim * 2, dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)  # (B,128,128)

        # 可选通道注意力
        if self.use_channel_ccc:
            self.ccc_512 = ChannelCCC(base_dim)
            self.ccc_128 = ChannelCCC(base_dim * 2)
        else:
            self.ccc_512 = nn.Identity()
            self.ccc_128 = nn.Identity()

        # 动态邻域 - 通道融合（在 512 / 128 尺度）
        if self.use_dynamic_fusion:
            # 512: 从原始 N 集合上以 idx_512 为中心，融合 enc0(N,64) -> (B,512,64)
            self.dyn512 = DynamicRadiusChannelFusion(
                in_channels=base_dim, out_channels=base_dim,
                num_neighbors=self.dyn_neighbors,
                min_radius=self.dyn_min_radius, max_radius=self.dyn_max_radius
            )
            # 128: 在 512 子集上以 idx_128 为中心，融合 feat_512(512,64) -> (B,128,128)
            self.dyn128 = DynamicRadiusChannelFusion(
                in_channels=base_dim, out_channels=base_dim * 2,
                num_neighbors=self.dyn_neighbors,
                min_radius=self.dyn_min_radius, max_radius=self.dyn_max_radius
            )
        else:
            self.dyn512 = None
            self.dyn128 = None

        # 位置编码
        self.pos512 = SharedMLP1D(3, base_dim * 2, dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)   # 3 -> 128
        self.posN   = SharedMLP1D(3, base_dim,     dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)   # 3 -> 64

        # 线性 GVA（在融合后的 512、N 尺度）
        if self.use_linear_gva:
            self.gva_512 = LinearSpatialGVA(dim=base_dim * 2)  # 作用在 fuse_512(128) 的通道数上
            self.gva_N   = LinearSpatialGVA(dim=base_dim * 2)  # 作用在 fuse_N(128)
        else:
            self.gva_512 = nn.Identity()
            self.gva_N   = nn.Identity()

        # Decoder 融合
        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1),  # 64 + 128 + 128 -> 128
            nn.BatchNorm1d(base_dim * 2, eps=self.bn_eps, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim * 2, 1),      # 128 + 64 + 64 -> 128
            nn.BatchNorm1d(base_dim * 2, eps=self.bn_eps, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        # DropPath
        self.branch_dp1 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()
        self.branch_dp2 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()

        # 分类头
        self.head = nn.Conv1d(base_dim * 2, num_classes, kernel_size=1)

        # 统一数值稳定的初始化
        self.apply(self._init_stable)

    # ---------- 初始化 ----------
    @staticmethod
    def _init_stable(m: nn.Module):
        """
        更稳的默认初始化：
        - Conv: Kaiming Normal (fan_out, relu)，bias=0
        - Linear: Xavier Normal，bias=0
        - BN: weight=1, bias=0（保守），其 eps/momentum 由构造时指定
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if getattr(m, "weight", None) is not None and m.weight is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------- 内部辅助 ----------
    def _sample_and_gather(self, xyz, feat, npoint):
        idx = farthest_point_sample(xyz, npoint)
        xyz_s = batched_index_points(xyz, idx)
        feat_s = batched_index_points(feat, idx)
        return xyz_s, feat_s, idx

    def _feature_group_dropout(self, points):
        if not self.training:
            return points
        out = points
        # RGB/Normal 按组丢弃；不改 shape
        if self.drop_rgb_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_rgb_p:
            out = out.clone()
            out[:, :, 3:6] = 0.0
        if self.drop_normal_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_normal_p:
            out = out.clone()
            out[:, :, 6:9] = 0.0
        return out

    # ---------- 前向 ----------
    def forward(self, points):
        """
        points: (B, N, 9) -> [xyz(0:3), rgb(3:6), normal(6:9)]
        return: logits (B, N, num_classes)
        """
        B, N, C = points.shape
        assert C == 9, f"Expect input features 9 (xyz+rgb+normal), but got {C}"

        if self.debug:
            print(f"[DEBUG] Input shape: {points.shape}")

        # 分组丢弃
        points = self._feature_group_dropout(points)

        # 拆分
        xyz = points[:, :, 0:3]
        xyz_normed = normalize_xyz(xyz)

        # 编码 0：输入到 64
        feat0 = self.enc0(points)  # (B,N,64)

        # 下采样到 512
        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0, npoint=512)  # (B,512,3),(B,512,64)
        enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)  # (B,512,67)
        feat_512 = self.enc512(enc_512_in)                    # (B,512,64)

        # —— 动态邻域-通道融合 @512：在 N 上用 idx_512 聚合 enc0(N,64) 到 512
        if self.use_dynamic_fusion and self.dyn512 is not None:
            dyn_512, _ = self.dyn512(points=xyz_normed, feats=feat0, center_idx=idx_512)  # (B,512,64)
            feat_512 = feat_512 + dyn_512  # 残差融合

        # 可选通道注意力
        feat_512 = self.ccc_512(feat_512)  # (B,512,64)

        # 再下采样到 128
        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)  # (B,128,3),(B,128,64)
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)  # (B,128,67)
        feat_128 = self.enc128(enc_128_in)                       # (B,128,128)

        # —— 动态邻域-通道融合 @128：在 512 子集上用 idx_128 聚合 feat_512(512,64) -> (B,128,128)
        if self.use_dynamic_fusion and self.dyn128 is not None:
            dyn_128, _ = self.dyn128(points=xyz_512, feats=feat_512, center_idx=idx_128)  # (B,128,128)
            feat_128 = feat_128 + dyn_128  # 残差融合

        # 可选通道注意力
        feat_128 = self.ccc_128(feat_128)  # (B,128,128)

        if self.debug:
            with torch.no_grad():
                n_nan1 = torch.isnan(feat_512).any().item()
                n_nan2 = torch.isnan(feat_128).any().item()
                print(f"[DEBUG] out1 shape: {feat_512.shape}, NaN={int(n_nan1)}, Inf={int(torch.isinf(feat_512).any().item())}")
                print(f"[DEBUG] out2 shape: {feat_128.shape}, NaN={int(n_nan2)}, Inf={int(torch.isinf(feat_128).any().item())}")

        # Decoder: 128 -> 512
        up128_to_512 = nearest_interpolate(xyz_512, xyz_128, feat_128)  # (B,512,128)
        pos512 = self.pos512(xyz_512)                                   # (B,512,128)

        # DropPath
        up128_to_512 = self.branch_dp1(up128_to_512)
        pos512 = self.branch_dp1(pos512)

        # 融合并（可选）加一层线性 GVA 做空间聚合
        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)  # (B,512,320)
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,128)
        fuse_512 = self.gva_512(fuse_512)  # 轻量空间注意

        # Decoder: 512 -> N
        up512_to_N = nearest_interpolate(xyz_normed, xyz_512, fuse_512)  # (B,N,128)
        posN = self.posN(xyz_normed)                                     # (B,N,64)

        up512_to_N = self.branch_dp2(up512_to_N)
        posN = self.branch_dp2(posN)

        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)            # (B,N,256)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,128)
        fuse_N = self.gva_N(fuse_N)  # 轻量空间注意

        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,K)

        if self.logit_temp and self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        if self.debug:
            with torch.no_grad():
                print(f"[DEBUG] logits shape: {logits.shape}")

        return logits

    # ---------- Loss（保持你原有稳定性守护） ----------
    def get_loss(
            self,
            points,
            labels,
            class_weights=None,
            ignore_index=-1,
            aux_weight=None,
            reduction='mean',
            label_smoothing: float = 0.0,
            focal_gamma: float = 0.0,
            **kwargs
    ):
        logits = self.forward(points)  # (B,N,K)

        # ===== Debug 1: 检查 logits =====
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[NaN Debug] NaN/Inf detected in logits, fixing...")
            logits = torch.nan_to_num(logits)

        B, N, K = logits.shape
        labels = labels.view(B, N)
        valid_mask = labels != ignore_index

        if valid_mask.sum() == 0:
            loss = logits.new_tensor(0.0, requires_grad=True)
            with torch.no_grad():
                acc = logits.new_tensor(0.0)
            stats = {"loss": float(loss.item()), "acc": float(acc.item()), "valid_points": 0}
            return loss, logits, stats

        logits_valid = logits[valid_mask]
        labels_valid = labels[valid_mask]

        # ===== Debug 2: 检查 logits_valid =====
        if torch.isnan(logits_valid).any() or torch.isinf(logits_valid).any():
            print("[NaN Debug] NaN/Inf detected in logits_valid, fixing...")
            logits_valid = torch.nan_to_num(logits_valid)

        weight = None
        if class_weights is not None:
            weight = (class_weights.to(logits.device, dtype=logits.dtype)
                      if isinstance(class_weights, torch.Tensor)
                      else torch.tensor(class_weights, dtype=logits.dtype, device=logits.device))

        log_probs = F.log_softmax(logits_valid, dim=-1)

        # ===== Debug 3: 检查 log_probs =====
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("[NaN Debug] NaN/Inf detected in log_probs, fixing...")
            log_probs = torch.nan_to_num(log_probs)

        probs = log_probs.exp()

        if label_smoothing is not None and label_smoothing > 0.0:
            eps = float(label_smoothing)
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(eps / K)
            true_dist.scatter_(1, labels_valid.unsqueeze(1), 1.0 - eps)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, labels_valid.unsqueeze(1), 1.0)

        ce = -(true_dist * log_probs).sum(dim=-1)

        if weight is not None:
            w = weight[labels_valid]
            ce = ce * w

        if focal_gamma is not None and focal_gamma > 0.0:
            gamma = float(focal_gamma)
            pt = (true_dist * probs).sum(dim=-1).clamp(min=1e-6, max=1.0)
            if torch.isnan(pt).any() or torch.isinf(pt).any():
                print("[NaN Debug] NaN/Inf detected in pt, fixing...")
                pt = torch.nan_to_num(pt)
            focal_weight = (1.0 - pt) ** gamma
            ce = focal_weight * ce

        if reduction == 'mean':
            loss_main = ce.mean()
        elif reduction == 'sum':
            loss_main = ce.sum()
        else:
            loss_main = ce

        loss = loss_main if aux_weight is None else loss_main * (1 - aux_weight)

        # ===== Debug 4: 检查 loss =====
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("[NaN Debug] NaN/Inf detected in final loss, fixing...")
            loss = torch.nan_to_num(loss)

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
