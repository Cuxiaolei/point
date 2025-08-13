# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # (x - y)^2 = x^2 + y^2 - 2xy
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
    B, Nt, _ = target_xyz.shape
    _, Ns, C = source_feat.shape
    # 距离 (B,Nt,Ns)
    dist = squared_distance(target_xyz, source_xyz)          # (B,Nt,Ns)
    idx = dist.argmin(dim=-1)                                # (B,Nt)
    # gather
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
    # 初始化：距离设为无穷大
    distances = torch.full((B, N), 1e10, device=device)
    # 随机选择一个初始点（也可以选质心最近的）
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        idx[:, i] = farthest
        # 计算到新选点的距离，并更新最小距离
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
    """
    def __init__(self, cin, cout, bn=True, act=True):
        super().__init__()
        layers = [nn.Conv1d(cin, cout, kernel_size=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(cout))
        if act:
            layers.append(nn.ReLU(inplace=True))
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
    def __init__(self, num_classes, base_dim=64, max_points=8000, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.max_points = max_points
        self.debug = debug

        # Encoder level-0 (N points)
        # 输入是 9 维：XYZ + RGB + Normal
        self.enc0 = SharedMLP1D(9, base_dim)  # -> (B,N,64)

        # 512 层的特征提炼：把 (XYZ_norm 3) + enc0_down(64) => 64
        self.enc512 = SharedMLP1D(3 + base_dim, base_dim)  # -> (B,512,64)

        # 128 层的特征提炼：把 (XYZ_norm 3) + enc512_down(64) => 128
        self.enc128 = SharedMLP1D(3 + base_dim, base_dim * 2)  # -> (B,128,128)

        # 位置编码（只用几何，避免把 RGB/Normal 混进位置）
        self.pos512 = SharedMLP1D(3, base_dim * 2)   # 3 -> 128
        self.posN   = SharedMLP1D(3, base_dim)       # 3 -> 64

        # Decoder 上采样整合
        # fuse_512 = cat([feat_512(64), up128_to_512(128), pos512(128)]) = 320
        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1),  # 64 + 128 + 128 = 320 -> 128
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True)
        )

        # fuse_N = cat([up512_to_N(128), enc0(64), posN(64)]) = 256
        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim * 2, 1),  # 128 + 64 + 64 = 256 -> 128
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True)
        )

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

    def forward(self, points):
        """
        points: (B, N, 9) -> [xyz(0:3), rgb(3:6), normal(6:9)]
        return: logits (B, N, num_classes)
        """
        B, N, C = points.shape
        assert C == 9, f"Expect input features 9 (xyz+rgb+normal), but got {C}"

        if self.debug:
            print(f"[DEBUG] Input shape: {points.shape}")

        # 拆分
        xyz = points[:, :, 0:3]                                 # (B,N,3)
        # rgb  = points[:, :, 3:6]
        # norm = points[:, :, 6:9]

        # 仅对 xyz 做归一化用于几何处理
        xyz_normed = normalize_xyz(xyz)                          # (B,N,3)

        # Encoder-0: 直接把 9 维输入嵌入到 64
        feat0 = self.enc0(points)                                # (B,N,64)

        # 下采样到 512
        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0, npoint=512)  # (B,512,3),(B,512,64)
        # 在 512 处再堆一个编码： [xyz_512(3) + feat0_512(64)] -> 64
        enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)     # (B,512,67)
        feat_512 = self.enc512(enc_512_in)                       # (B,512,64)

        # 再下采样到 128
        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)  # (B,128,3),(B,128,64)
        # 在 128 处编码： [xyz_128(3) + feat_512_128(64)] -> 128
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)  # (B,128,67)
        feat_128 = self.enc128(enc_128_in)                       # (B,128,128)

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

        # 融合（确保 64 + 128 + 128 = 320）
        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)  # (B,512,320)
        # 走 up1
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,128)

        # Decoder: 512 -> N
        up512_to_N = nearest_interpolate(xyz_normed, xyz_512, fuse_512)  # (B,N,128)

        # N 的位置编码
        posN = self.posN(xyz_normed)                                    # (B,N,64)

        # 融合（确保 128 + 64 + 64 = 256）
        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)           # (B,N,256)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,128)

        # 分类头
        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,num_classes)

        if self.debug:
            with torch.no_grad():
                print(f"[DEBUG] logits shape: {logits.shape}")

        return logits

    def get_loss(self, points, labels, class_weights=None, ignore_index=-1, aux_weight=None, reduction='mean', **kwargs):
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

        logits_valid = logits[valid_mask]
        labels_valid = labels[valid_mask]

        weight = None
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weight = class_weights.to(logits.device, dtype=logits.dtype)
            else:
                weight = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)

        loss_main = F.cross_entropy(
            logits_valid,
            labels_valid,
            weight=weight,
            reduction=reduction
        )

        # 预留辅助损失接口（即使现在不用，也不会报错）
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
