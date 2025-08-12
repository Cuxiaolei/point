import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance(a, b):
    """计算点云之间的平方欧氏距离"""
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)  # (B, M, 1)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
    inner = torch.bmm(a, b.transpose(1, 2))  # (B, M, N)
    return a_sq + b_sq - 2 * inner


def safe_knn_idx(dist, radius, num_neighbors, num_points):
    """计算 KNN 索引时，确保索引不越界。"""
    radius = torch.tensor(radius, device=dist.device)  # 确保 radius 是 tensor 类型
    mask = dist <= radius.unsqueeze(-1)  # (B, M, N)
    masked_dist = dist.clone()
    masked_dist[~mask] = float('inf')  # 非有效索引处设置为无穷大

    # 打印调试信息
    print(f"[safe_knn_idx] dist min: {dist.min()}, dist max: {dist.max()}")
    print(f"[safe_knn_idx] masked_dist min: {masked_dist.min()}, masked_dist max: {masked_dist.max()}")

    _, knn_idx = torch.topk(masked_dist, k=num_neighbors, dim=-1, largest=False)

    # 打印 knn_idx 的值和维度
    print(f"[safe_knn_idx] knn_idx shape: {knn_idx.shape}")
    print(f"[safe_knn_idx] knn_idx: {knn_idx[0]}")  # 打印第一个样本的 knn_idx

    # 确保索引不超出维度
    knn_idx = knn_idx.clamp(max=num_points - 1)
    return knn_idx

class DynamicRadiusChannelFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors=32, min_radius=0.05, max_radius=0.3):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 通道注意力机制
        self.channel_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        # 融合后的降维
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, points, feats, center_idx):
        B, N, _ = points.shape
        M = center_idx.shape[1]

        # 获取中心点的坐标 (B, M, 3)
        centers = torch.gather(points, 1, center_idx.unsqueeze(-1).repeat(1, 1, 3))

        # 计算到所有点的距离 (B, M, N)
        dist2 = pairwise_distance(centers, points).clamp(min=0)
        dist = torch.sqrt(dist2 + 1e-8)

        # 打印每一步的维度以调试
        print(f"[DynamicRadiusChannelFusion] centers shape: {centers.shape}")  # (B, M, 3)
        print(f"[DynamicRadiusChannelFusion] dist shape: {dist.shape}")  # (B, M, N)

        # 计算有效邻域索引
        knn_idx = safe_knn_idx(dist, self.max_radius, self.num_neighbors, N)

        # 确保 knn_idx 的维度正确
        print(f"[DynamicRadiusChannelFusion] knn_idx shape: {knn_idx.shape}")
        print(f"[DynamicRadiusChannelFusion] knn_idx: {knn_idx[0]}")  # 打印第一个样本的 knn_idx

        # 获取邻域特征
        batch_inds = torch.arange(B, device=points.device).view(B, 1, 1).repeat(1, M, self.num_neighbors)
        neigh_feats = feats[batch_inds, knn_idx]  # (B, M, K, C)
        centers_feats = torch.gather(feats, 1, center_idx.unsqueeze(-1).repeat(1, 1, self.in_ch))  # (B, M, C)
        centers_feats_exp = centers_feats.unsqueeze(2).repeat(1, 1, self.num_neighbors, 1)  # (B,M,K,C)

        # 通道融合：对于每邻域，计算中心与邻域的通道特征融合
        combo = torch.cat([centers_feats_exp, neigh_feats], dim=-1)  # (B,M,K,2C)
        combo2 = combo.view(B * M * self.num_neighbors, -1)
        channel_w = self.channel_proj(combo2)  # (B*M*K, C)
        channel_w = channel_w.view(B, M, self.num_neighbors, self.in_ch)

        # 使用通道权重调整邻域特征
        fused_nei = (neigh_feats * channel_w).mean(dim=2)  # (B, M, C)

        # 融合中心特征
        fused = fused_nei + centers_feats  # (B, M, C)
        out = self.mlp(fused)  # (B, M, out_ch)
        print(f"[DynamicRadiusChannelFusion] out shape: {out.shape}")  # 打印输出的维度
        return out, knn_idx


class ChannelCCC(nn.Module):
    """
    受 CAA 的 CCC 子模块启发：计算通道关联（C x C 相似度），并做轻量重构。
    输入: (B, N, C) -> 输出: (B, N, C)
    """

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(dim // 4, 8)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        ch_desc = x.mean(dim=1)  # (B, C)
        h = F.relu(self.linear1(ch_desc))
        att = torch.sigmoid(self.linear2(h)).unsqueeze(1)  # (B, 1, C)
        return self.norm(x * att + x)  # 用注意力加权通道特征


class LinearSpatialGVA(nn.Module):
    """
    线性化的空间聚合：采用类似线性注意力的键值编码（避免全局 Softmax）
    输入: x (B, N, C)
    输出: y (B, N, C)
    """

    def __init__(self, dim, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.q = nn.Linear(dim, kv_dim)  # 查询映射
        self.k = nn.Linear(dim, kv_dim)  # 键映射
        self.v = nn.Linear(dim, dim)  # 值映射

    def forward(self, x):
        B, N, C = x.shape
        Q = self.q(x)  # (B, N, kv_dim)
        K = self.k(x)  # (B, N, kv_dim)
        V = self.v(x)  # (B, N, C)

        KV = torch.einsum('bnd,bnc->bdc', K, V)  # (B, kv_dim, C)
        out = torch.einsum('bnd,bdc->bnc', Q, KV)  # (B, N, C)

        Ksum = K.sum(dim=1, keepdim=True)  # (B, 1, kv_dim)
        denom = torch.einsum('bnd,bd->bn', Q, Ksum.squeeze(1)).unsqueeze(-1).clamp(min=1e-6)
        out = out / denom  # 归一化
        return out
