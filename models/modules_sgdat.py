import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
import random
from config import Config

def pairwise_distance(a, b):
    """(B, M, 3), (B, N, 3) -> (B, M, N) squared distances"""
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)  # (B,M,1)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True).transpose(1,2)  # (B,1,N)
    inner = torch.bmm(a, b.transpose(1,2))  # (B,M,N)
    return a_sq + b_sq - 2 * inner

def farthest_point_sample(xyz, npoint):
    """Simple (but vectorized-ish) FPS. xyz: (B,N,3) -> idx: (B,npoint)"""
    device = xyz.device
    B, N, _ = xyz.shape
    if npoint >= N:
        idx = torch.arange(N, device=device).unsqueeze(0).repeat(B,1)
        return idx[:, :npoint]
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B,1,3)
        dist = ((xyz - centroid_xyz) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids

def safe_knn_idx(dist, radius, num_neighbors, num_points):
    """dist: (B,M,N); radius scalar or tensor -> return (B,M,K) indices"""
    device = dist.device
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device)
    mask = dist <= radius.unsqueeze(-1)
    masked = dist.clone()
    masked[~mask] = 1e9
    masked = masked.clamp(min=0.0)
    K = min(num_neighbors, num_points)
    _, idx = torch.topk(masked, k=K, dim=-1, largest=False)
    idx = idx.clamp(0, num_points-1)
    return idx

class DynamicRadiusChannelFusion(nn.Module):
    """
    Dynamic radius neighborhood + channel fusion.
    Expects:
      points: (B,N,3)
      feats:  (B,N,C)
      center_idx: (B,M) long
    Returns:
      out: (B,M,out_ch)
      knn_idx: (B,M,K)
    """
    def __init__(self, in_channels, out_channels, num_neighbors=16, min_radius=0.02, max_radius=0.3):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.in_ch = in_channels
        self.out_ch = out_channels

        self.pre_norm = nn.LayerNorm(in_channels * 2)

        self.channel_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        )
        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()

    def forward(self, points, feats, center_idx):
        B, N, _ = points.shape
        M = center_idx.shape[1]
        centers = torch.gather(points, 1, center_idx.unsqueeze(-1).repeat(1,1,3))
        dist2 = pairwise_distance(centers, points).clamp(min=0.0)
        dist = torch.sqrt(dist2 + 1e-8)

        knn_idx = safe_knn_idx(dist, self.max_radius, self.num_neighbors, N)

        K = knn_idx.size(-1)
        batch_inds = torch.arange(B, device=points.device).view(B,1,1).expand(-1, M, K)
        neigh_feats = feats[batch_inds, knn_idx]
        centers_feats = torch.gather(feats, 1, center_idx.unsqueeze(-1).repeat(1,1,self.in_ch))
        centers_feats_exp = centers_feats.unsqueeze(2).expand(-1, -1, K, -1)

        combo = torch.cat([centers_feats_exp, neigh_feats], dim=-1)
        combo_flat = combo.view(-1, combo.size(-1))
        combo_norm = self.pre_norm(combo_flat)
        channel_w = self.channel_proj(combo_norm).view(B, M, K, self.in_ch)

        weighted = (neigh_feats * channel_w).mean(dim=2)
        fused = weighted + centers_feats
        if Config.ENABLE_FEATURE_DROP:
            fused = feature_dropout(fused, Config.FEATURE_DROP_PROB)
        out = self.mlp(fused)
        out = self.drop_path(out)
        return out, knn_idx

class ChannelCCC(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(dim//4, 8)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()

    def forward(self, x):
        ch_desc = x.mean(dim=1)
        h = F.relu(self.linear1(ch_desc))
        att = torch.sigmoid(self.linear2(h)).unsqueeze(1)
        out = self.norm(x * att + x)
        if Config.ENABLE_FEATURE_DROP:
            out = feature_dropout(out, Config.FEATURE_DROP_PROB)
        out = self.dropout(out)
        out = self.drop_path(out)
        return out

class LinearSpatialGVA(nn.Module):
    """Linearized spatial aggregation with normalization to prevent explosion."""
    def __init__(self, dim, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, kv_dim)
        self.k = nn.Linear(dim, kv_dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = Dropout(Config.DROPOUT_RATE) if Config.ENABLE_DROPOUT else nn.Identity()
        self.drop_path = DropPath(Config.DROPPATH_PROB) if Config.ENABLE_DROPPATH else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        KV = torch.einsum('bnd,bnc->bdc', K, V) / max(1, N)
        out = torch.einsum('bnd,bdc->bnc', Q, KV)

        Ksum = K.sum(dim=1, keepdim=True)
        denom = (Q * Ksum).sum(dim=-1, keepdim=True).clamp(min=1e-3)

        out = out / denom
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        if Config.ENABLE_FEATURE_DROP:
            out = feature_dropout(out, Config.FEATURE_DROP_PROB)
        out = self.dropout(out)
        out = self.drop_path(out)
        return out

# -----------------------------
# DropPath / Stochastic Depth
# -----------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
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
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# -----------------------------
# Feature Dropout
# -----------------------------
def feature_dropout(x, drop_prob=0.1):
    if drop_prob > 0 and x.requires_grad and random.random() < drop_prob:
        drop_mask = torch.rand_like(x) > drop_prob
        return x * drop_mask
    return x
