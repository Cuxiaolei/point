# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容导入（允许作为包或脚本）
try:
    from .modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath
except Exception:
    from models.modules_sgdat import ChannelCCC, LinearSpatialGVA, DropPath

# 引入配置
from config import Config


# -----------------------------
# Utils
# -----------------------------
def normalize_xyz(xyz, eps=1e-6):
    center = xyz.mean(dim=1, keepdim=True)
    xyz_centered = xyz - center
    scale = torch.sqrt((xyz_centered ** 2).sum(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]) + eps
    xyz_normed = xyz_centered / scale
    return xyz_normed


def batched_index_points(points, idx):
    B, N, C = points.shape
    _, M = idx.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1).repeat(1, M)
    return points[batch_indices, idx, :]


def squared_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    xx = (src ** 2).sum(dim=-1, keepdim=True)
    yy = (dst ** 2).sum(dim=-1).unsqueeze(1)
    xy = torch.bmm(src, dst.transpose(1, 2))
    dist = xx + yy - 2 * xy
    dist = torch.clamp(dist, min=0.0)
    return dist


def nearest_interpolate(target_xyz, source_xyz, source_feat):
    dist = squared_distance(target_xyz, source_xyz)
    idx = dist.argmin(dim=-1)
    feat = batched_index_points(source_feat, idx)
    return feat


@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)

    return idx


# -----------------------------
# Basic Blocks
# -----------------------------
class SharedMLP1D(nn.Module):
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
        x = x.permute(0, 2, 1).contiguous()
        x = self.net(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


# -----------------------------
# SGDAT
# -----------------------------
class SGDAT(nn.Module):
    def __init__(
        self,
        num_classes,
        base_dim=64,
        max_points=8000,
        debug=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim
        self.max_points = max_points
        self.debug = debug

        # 从 config.py 读取抗过拟合参数
        self.dropout_p = Config.DROPOUT_RATE if Config.ENABLE_DROPOUT else 0.0
        self.droppath_prob = Config.DROPPATH_PROB if Config.ENABLE_DROPPATH else 0.0
        self.drop_rgb_p = Config.DROP_RGB_PROB if Config.ENABLE_FEATURE_DROP else 0.0
        self.drop_normal_p = Config.DROP_NORMAL_PROB if Config.ENABLE_FEATURE_DROP else 0.0
        self.logit_temp = Config.LOGIT_TEMP
        self.use_channel_ccc = Config.ENABLE_CHANNEL_CCC

        # Encoder level-0
        self.enc0 = SharedMLP1D(9, base_dim, dropout=self.dropout_p)

        self.enc512 = SharedMLP1D(3 + base_dim, base_dim, dropout=self.dropout_p)
        self.enc128 = SharedMLP1D(3 + base_dim, base_dim * 2, dropout=self.dropout_p)

        if self.use_channel_ccc:
            self.ccc_512 = ChannelCCC(base_dim)
            self.ccc_128 = ChannelCCC(base_dim * 2)
        else:
            self.ccc_512 = nn.Identity()
            self.ccc_128 = nn.Identity()

        self.pos512 = SharedMLP1D(3, base_dim * 2, dropout=self.dropout_p)
        self.posN   = SharedMLP1D(3, base_dim, dropout=self.dropout_p)

        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1),
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim * 2, 1),
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        self.branch_dp1 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()
        self.branch_dp2 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()

        self.head = nn.Conv1d(base_dim * 2, num_classes, kernel_size=1)

    def _sample_and_gather(self, xyz, feat, npoint):
        idx = farthest_point_sample(xyz, npoint)
        xyz_s = batched_index_points(xyz, idx)
        feat_s = batched_index_points(feat, idx)
        return xyz_s, feat_s, idx

    def _feature_group_dropout(self, points):
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
        B, N, C = points.shape
        assert C == 9, f"Expect input features 9 (xyz+rgb+normal), but got {C}"

        if self.debug:
            print(f"[DEBUG] Input shape: {points.shape}")

        points = self._feature_group_dropout(points)
        xyz = points[:, :, 0:3]
        xyz_normed = normalize_xyz(xyz)

        feat0 = self.enc0(points)

        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0, npoint=512)
        enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)
        feat_512 = self.enc512(enc_512_in)
        feat_512 = self.ccc_512(feat_512)

        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)
        feat_128 = self.enc128(enc_128_in)
        feat_128 = self.ccc_128(feat_128)

        if self.debug:
            with torch.no_grad():
                print(f"[DEBUG] out1 shape: {feat_512.shape}")
                print(f"[DEBUG] out2 shape: {feat_128.shape}")

        up128_to_512 = nearest_interpolate(xyz_512, xyz_128, feat_128)
        pos512 = self.pos512(xyz_512)

        up128_to_512 = self.branch_dp1(up128_to_512)
        pos512 = self.branch_dp1(pos512)

        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        up512_to_N = nearest_interpolate(xyz_normed, xyz_512, fuse_512)
        posN = self.posN(xyz_normed)

        up512_to_N = self.branch_dp2(up512_to_N)
        posN = self.branch_dp2(posN)

        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        if self.logit_temp and self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        return logits

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
        logits = self.forward(points)
        B, N, K = logits.shape
        labels = labels.view(B, N)

        valid_mask = labels != ignore_index
        if valid_mask.sum() == 0:
            loss = logits.new_tensor(0.0, requires_grad=True)
            with torch.no_grad():
                acc = logits.new_tensor(0.0)
            return loss, logits, {"loss": float(loss), "acc": float(acc), "valid_points": 0}

        logits_valid = logits[valid_mask]
        labels_valid = labels[valid_mask]

        weight = None
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weight = class_weights.to(logits.device, dtype=logits.dtype)
            else:
                weight = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)

        log_probs = F.log_softmax(logits_valid, dim=-1)
        probs = log_probs.exp()

        if label_smoothing and label_smoothing > 0.0:
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

        if focal_gamma and focal_gamma > 0.0:
            gamma = float(focal_gamma)
            pt = (true_dist * probs).sum(dim=-1).clamp(min=1e-6, max=1.0)
            focal_weight = (1.0 - pt) ** gamma
            ce = focal_weight * ce

        if reduction == 'mean':
            loss_main = ce.mean()
        elif reduction == 'sum':
            loss_main = ce.sum()
        else:
            loss_main = ce

        if aux_weight:
            loss = loss_main * (1 - aux_weight)
        else:
            loss = loss_main

        with torch.no_grad():
            pred = logits_valid.argmax(dim=-1)
            correct = (pred == labels_valid).sum()
            acc = correct.float() / labels_valid.numel()

        stats = {"loss": float(loss), "acc": float(acc), "valid_points": int(valid_mask.sum().item())}
        return loss, logits, stats
