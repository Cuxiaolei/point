import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
import random
from config import Config

# =========================
# Common helpers (stable)
# =========================
_STABLE_EPS = 1e-6
_LN_EPS = 1e-5


def init_weights_stable(m: nn.Module):
    """Stable default init: Linear -> Xavier; LayerNorm -> (1,0)."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute squared pairwise euclidean distances in a numerically stable way.
      a: (B, M, 3), b: (B, N, 3) -> (B, M, N)
    """
    # Convert to float32 for stability if in mixed precision
    dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a_ = a.to(dtype)
    b_ = b.to(dtype)
    a_sq = (a_ ** 2).sum(dim=-1, keepdim=True)              # (B, M, 1)
    b_sq = (b_ ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
    inner = torch.bmm(a_, b_.transpose(1, 2))               # (B, M, N)
    dist2 = a_sq + b_sq - 2 * inner
    dist2 = torch.clamp(dist2, min=0.0)                     # avoid tiny negatives
    dist2 = torch.nan_to_num(dist2)
    return dist2


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Simple (but vectorized-ish) FPS. xyz: (B,N,3) -> idx: (B,npoint)"""
    device = xyz.device
    B, N, _ = xyz.shape
    if npoint >= N:
        idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        return idx[:, :npoint]
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid_xyz) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def safe_knn_idx(dist: torch.Tensor,
                 radius,
                 num_neighbors: int,
                 num_points: int) -> torch.Tensor:
    """
    dist: (B, M, N) pairwise distances (non-negative)
    Return (B, M, K) indices within radius, with safe fallbacks.
    """
    device = dist.device
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device, dtype=dist.dtype)
    # allow broadcasting if radius is scalar
    mask = dist <= radius.unsqueeze(-1)
    # mask invalid distances to large number, then topk
    masked = torch.where(mask, dist, torch.full_like(dist, 1e9))
    masked = masked.clamp(min=0.0)
    K = min(num_neighbors, num_points)
    # If all are 1e9 (no neighbor), topk will still return some indices, that's OK
    _, idx = torch.topk(masked, k=K, dim=-1, largest=False)  # (B,M,K)
    idx = idx.clamp(0, num_points - 1)
    return idx


# -----------------------------
# Stochastic Depth / DropPath
# -----------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------
# Feature Dropout
# -----------------------------
def feature_dropout(x: torch.Tensor, drop_prob: float = 0.1):
    """
    Per-element feature dropout with runtime gating. Returns masked x.
    """
    if drop_prob <= 0.0:
        return x
    # Make the mask deterministic w.r.t. autograd graph size
    mask = (torch.rand_like(x) > drop_prob).to(x.dtype)
    return x * mask


# -----------------------------
# Residual Scale (learnable)
# -----------------------------
class ResidualScale(nn.Module):
    """
    Learnable scalar (single parameter) to scale residual branch.
    Initialized to init_value (e.g., 0.5) to improve early-stage stability.
    """
    def __init__(self, init_value: float = 0.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor):
        return x * self.scale


# ======================================
# Core modules (stabilized implementations)
# ======================================
class DynamicRadiusChannelFusion(nn.Module):
    """
    Dynamic radius neighborhood + channel fusion (stabilized).
    Expects:
      points: (B,N,3)
      feats:  (B,N,C)
      center_idx: (B,M) long
    Returns:
      out: (B,M,out_ch)
      knn_idx: (B,M,K)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_neighbors: int = 16,
                 min_radius: float = 0.02,
                 max_radius: float = 0.3):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)

        # pre/post norms for stability
        self.pre_norm = nn.LayerNorm(in_channels * 2, eps=_LN_EPS)
        self.post_norm = nn.LayerNorm(out_channels, eps=_LN_EPS)

        # channel attention (sigmoid gating)
        self.channel_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        # local fusion + projection
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        )

        # residual scaling to tame magnitude growth
        self.res_scale = ResidualScale(init_value=0.5)

        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()

        # init
        self.apply(init_weights_stable)

    def forward(self,
                points: torch.Tensor,
                feats: torch.Tensor,
                center_idx: torch.Tensor):
        """
        points: (B,N,3), feats: (B,N,C), center_idx: (B,M)
        """
        B, N, _ = points.shape
        M = center_idx.shape[1]

        centers = torch.gather(points, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,M,3)
        dist2 = pairwise_distance(centers, points)                                     # (B,M,N)
        dist = torch.sqrt(dist2 + _STABLE_EPS)                                         # (B,M,N)

        # dynamic KNN within max_radius (min_radius currently unused but kept for future)
        knn_idx = safe_knn_idx(dist, self.max_radius, self.num_neighbors, N)           # (B,M,K)
        K = knn_idx.size(-1)

        # gather neighbor feats
        batch_inds = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, M, K)
        neigh_feats = feats[batch_inds, knn_idx]                                       # (B,M,K,C)
        neigh_feats = torch.nan_to_num(neigh_feats)

        # center feats
        centers_feats = torch.gather(feats, 1, center_idx.unsqueeze(-1).expand(-1, -1, self.in_ch))  # (B,M,C)
        centers_feats = torch.nan_to_num(centers_feats)

        # channel fusion weights from [center || neighbor]
        centers_feats_exp = centers_feats.unsqueeze(2).expand(-1, -1, K, -1)          # (B,M,K,C)
        combo = torch.cat([centers_feats_exp, neigh_feats], dim=-1)                    # (B,M,K,2C)
        combo_flat = combo.reshape(-1, combo.size(-1))                                 # (B*M*K,2C)
        combo_norm = self.pre_norm(combo_flat)
        channel_w = self.channel_proj(combo_norm).view(B, M, K, self.in_ch)            # (B,M,K,C)

        # fuse neighbors (mean with channel gating), residual to center
        weighted = (neigh_feats * channel_w).mean(dim=2)                               # (B,M,C)
        weighted = torch.nan_to_num(weighted)
        fused = centers_feats + self.res_scale(weighted)                               # (B,M,C)

        if Config.ENABLE_FEATURE_DROP:
            fused = feature_dropout(fused, Config.FEATURE_DROP_PROB)

        out = self.mlp(fused)                                                          # (B,M,out_ch)
        out = torch.nan_to_num(out)
        out = self.post_norm(out)
        out = self.drop_path(out)
        return out, knn_idx


class ChannelCCC(nn.Module):
    """
    Channel-wise compact context (CCC) with residual & normalization.
    """
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(dim // 4, 8)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim, eps=_LN_EPS)
        self.dropout = Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()
        self.res_scale = ResidualScale(init_value=0.5)

        self.apply(init_weights_stable)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C)
        """
        # global channel descriptor (avg over N)
        ch_desc = x.mean(dim=1)                            # (B, C)
        h = F.relu(self.linear1(ch_desc), inplace=True)    # (B, hidden)
        att = torch.sigmoid(self.linear2(h)).unsqueeze(1)  # (B,1,C)
        att = torch.nan_to_num(att)

        y = x * att                                        # (B,N,C)
        y = self.res_scale(y) + x                          # residual with scale
        y = self.norm(y)

        if Config.ENABLE_FEATURE_DROP:
            y = feature_dropout(y, Config.FEATURE_DROP_PROB)
        y = self.dropout(y)
        y = self.drop_path(y)
        return y


class LinearSpatialGVA(nn.Module):
    """
    Linearized spatial aggregation with normalization to prevent explosion.
    Q, K, V are linear projections; aggregation uses KV cache and a safe denom.
    """
    def __init__(self, dim: int, kv_dim: int | None = None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, kv_dim)
        self.k = nn.Linear(dim, kv_dim)
        self.v = nn.Linear(dim, dim)

        self.dropout = Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()
        self.post_norm = nn.LayerNorm(dim, eps=_LN_EPS)
        self.res_scale = ResidualScale(init_value=0.5)

        self.apply(init_weights_stable)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        # project
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        Q = torch.nan_to_num(Q)
        K = torch.nan_to_num(K)
        V = torch.nan_to_num(V)

        # KV cache (scale by N to keep magnitude)
        # (bnd,bnc->bdc) / N
        KV = torch.einsum('bnd,bnc->bdc', K, V) / max(1, N)
        KV = torch.nan_to_num(KV)

        # aggregate: (bnd,bdc->bnc)
        out = torch.einsum('bnd,bdc->bnc', Q, KV)
        out = torch.nan_to_num(out)

        # denom = <Q, sum(K)> (per token), lower-bounded
        Ksum = K.sum(dim=1, keepdim=True)                               # (B,1,Dk)
        denom = (Q * Ksum).sum(dim=-1, keepdim=True).abs()              # (B,N,1)
        denom = torch.clamp(denom, min=_STABLE_EPS)

        out = out / denom
        out = torch.nan_to_num(out)

        if Config.ENABLE_FEATURE_DROP:
            out = feature_dropout(out, Config.FEATURE_DROP_PROB)

        # residual + norm
        out = self.res_scale(out) + x
        out = self.post_norm(out)

        out = self.dropout(out)
        out = self.drop_path(out)
        return out
