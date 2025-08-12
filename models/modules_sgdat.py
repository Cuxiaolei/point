# models/modules_sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(a, b):
    """Compute squared Euclidean distance between two point sets.
    a: (B, M, 3), b: (B, N, 3) -> (B, M, N)
    """
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)  # (B, M, 1)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
    inner = torch.bmm(a, b.transpose(1, 2))  # (B, M, N)
    return a_sq + b_sq - 2 * inner


def farthest_point_sample(xyz, npoint):
    """Farthest Point Sampling (FPS)
    xyz: (B, N, 3)
    return idx: (B, npoint) (long)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # pick random initial index
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B,1,3)
        dist = ((xyz - centroid_xyz) ** 2).sum(-1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def safe_knn_idx(dist, radius, num_neighbors, num_points):
    """Safe KNN: given dist (B, M, N) and radius scalar, return knn indices (B, M, K)
    - If a point has fewer than K candidates within radius, we still pick smallest distances by topk (with large value handling).
    """
    device = dist.device
    # radius may be tensor or float
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, device=device)

    # mask of valid neighbors within radius
    mask = dist <= radius.unsqueeze(-1)  # (B, M, N) broadcasting
    masked = dist.clone()
    # put invalid entries to large value (so topk chooses the true smallest distances)
    masked[~mask] = 1e9

    # Avoid potential exact zeros messing up topk: no need but keep numerically stable
    masked = masked.clamp(min=0.0)

    # topk smallest distances along last dim
    # If num_neighbors > N, topk will duplicate indices; we clamp K <= N
    K = min(num_neighbors, num_points)
    _, idx = torch.topk(masked, k=K, dim=-1, largest=False)  # (B, M, K)

    # final clamp safety (ensure indices in [0, N-1])
    idx = idx.clamp(min=0, max=num_points - 1)
    return idx


class DynamicRadiusChannelFusion(nn.Module):
    """
    Dynamic neighborhood selection (radius-based) + channel fusion.
    Inputs:
      - points: (B, N, 3)
      - feats:  (B, N, C)
      - center_idx: (B, M) long indices into N
    Returns:
      - out: (B, M, out_ch)
      - knn_idx: (B, M, K)
    """
    def __init__(self, in_channels, out_channels, num_neighbors=32, min_radius=0.02, max_radius=0.3):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.in_ch = in_channels
        self.out_ch = out_channels

        # light channel weighting (inspired by CCC)
        self.channel_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, points, feats, center_idx):
        """
        points: (B,N,3)
        feats: (B,N,C)
        center_idx: (B,M)  long indices
        """
        B, N, _ = points.shape
        # validate center_idx
        if center_idx.dim() != 2:
            raise RuntimeError(f"center_idx must be (B, M), got {center_idx.shape}")

        M = center_idx.shape[1]

        # gather center coordinates (B, M, 3)
        centers = torch.gather(points, 1, center_idx.unsqueeze(-1).repeat(1, 1, 3))
        # pairwise distance (B, M, N)
        dist2 = pairwise_distance(centers, points).clamp(min=0.0)
        dist = torch.sqrt(dist2 + 1e-8)

        # compute knn indices within radius (safe)
        knn_idx = safe_knn_idx(dist, self.max_radius, self.num_neighbors, N)  # (B,M,K)

        # debug prints (shallow)
        # print shapes
        # print(f"[DRCF] centers {centers.shape}, dist {dist.shape}, knn_idx {knn_idx.shape}")

        # gather neighbor features
        batch_inds = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, M, knn_idx.size(-1))  # (B,M,K)
        neigh_feats = feats[batch_inds, knn_idx]  # (B, M, K, C)

        centers_feats = torch.gather(feats, 1, center_idx.unsqueeze(-1).repeat(1, 1, self.in_ch))  # (B, M, C)
        centers_feats_exp = centers_feats.unsqueeze(2).expand(-1, -1, knn_idx.size(-1), -1)  # (B,M,K,C)

        # channel fusion: concat center & neighbor in channel dim -> compute channel weights
        combo = torch.cat([centers_feats_exp, neigh_feats], dim=-1)  # (B,M,K,2C)
        combo_flat = combo.view(-1, combo.size(-1))  # (B*M*K, 2C)
        channel_w = self.channel_proj(combo_flat).view(B, M, knn_idx.size(-1), self.in_ch)  # (B,M,K,C)

        # weighted average of neighbor feats (per channel)
        weighted = (neigh_feats * channel_w).mean(dim=2)  # (B, M, C)

        fused = weighted + centers_feats  # residual add
        out = self.mlp(fused)  # (B, M, out_ch)
        return out, knn_idx


class ChannelCCC(nn.Module):
    """Lightweight channel attention inspired by CCC."""
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(dim // 4, 8)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C)
        ch_desc = x.mean(dim=1)  # (B, C)
        h = F.relu(self.linear1(ch_desc))
        att = torch.sigmoid(self.linear2(h)).unsqueeze(1)  # (B,1,C)
        return self.norm(x * att + x)


class LinearSpatialGVA(nn.Module):
    """
    Linearized spatial aggregation (lightweight GVA-like).
    x: (B,N,C)
    """
    def __init__(self, dim, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, kv_dim)
        self.k = nn.Linear(dim, kv_dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        Q = self.q(x)  # (B,N,kv)
        K = self.k(x)  # (B,N,kv)
        V = self.v(x)  # (B,N,C)

        KV = torch.einsum('bnd,bnc->bdc', K, V)  # (B,kv,C)
        out = torch.einsum('bnd,bdc->bnc', Q, KV)  # (B,N,C)

        Ksum = K.sum(dim=1, keepdim=True)  # (B,1,kv)
        denom = (Q * Ksum).sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B,N,1)
        out = out / denom
        return out
