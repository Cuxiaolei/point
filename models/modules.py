import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicBallQuery(nn.Module):
    """动态球查询模块（修复索引越界，适配通道变化）"""

    def __init__(self, min_radius=0.05, max_radius=0.3, num_neighbors=16):
        super().__init__()
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors

    def forward(self, points, features, center_indices):
        B, N, _ = points.shape  # (B, N, 3)
        M = center_indices.shape[1]  # 中心点数量
        C = features.shape[-1]  # 特征通道数（动态适配）

        # 安全检查：确保中心点索引在有效范围内
        assert (center_indices >= 0).all() and (center_indices < N).all(), \
            f"中心点索引越界：有效范围[0, {N - 1}]，实际出现{center_indices[(center_indices < 0) | (center_indices >= N)].unique()}"

        # 获取中心点坐标 (B, M, 3)
        centers = torch.gather(
            points,
            dim=1,
            index=center_indices.unsqueeze(-1).repeat(1, 1, 3)
        )

        # 分块计算距离（减少内存占用）
        dist = []
        block_size = 1024  # 每次处理1024个中心点
        for b in range(B):
            batch_dist = []
            for i in range(0, M, block_size):
                current_centers = centers[b, i:i + block_size]  # (K, 3)
                current_dist = torch.norm(
                    points[b].unsqueeze(0) - current_centers.unsqueeze(1),  # (K, N, 3)
                    dim=-1
                )
                batch_dist.append(current_dist)
            dist.append(torch.cat(batch_dist, dim=0))
        dist = torch.stack(dist, dim=0)  # (B, M, N)

        # 动态半径计算（添加数值稳定性处理）
        init_mask = dist < self.min_radius
        init_counts = init_mask.sum(dim=-1, keepdim=True).float()
        # 避免除以零：当最小半径为0时的特殊处理
        if self.min_radius == 0:
            density = torch.zeros_like(init_counts)
        else:
            density = init_counts / ((4 / 3) * np.pi * self.min_radius ** 3 + 1e-8)
        # 避免密度为0导致的除零
        density_max = density.max() + 1e-8
        radii = self.min_radius + (self.max_radius - self.min_radius) * (1 - density / density_max)

        # 筛选邻域点并限制索引范围
        mask = dist < radii
        dist = dist.masked_fill(~mask, 1e10)
        _, knn_indices = torch.topk(dist, k=self.num_neighbors, dim=-1, largest=False)

        # 确保索引不越界
        max_valid_index = features.shape[1] - 1
        knn_indices = torch.clamp(knn_indices, 0, max_valid_index)

        # 提取邻域特征（动态适配特征通道）
        batch_indices = torch.arange(B, device=features.device).unsqueeze(1).unsqueeze(2).repeat(1, M,
                                                                                                 self.num_neighbors)
        neighbor_features = features[batch_indices, knn_indices]  # (B, M, K, C)

        return neighbor_features


class AdaptiveAggregation(nn.Module):
    """自适应聚合层（确保输入输出通道匹配）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 卷积层输入通道严格匹配in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 注意力层输入通道匹配
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(out_channels)  # 稳定训练

    def forward(self, x):
        """
        Args:
            x: (B, M, K, C) 其中C必须等于in_channels
        """
        # 断言检查输入通道（提前发现不匹配）
        assert x.shape[-1] == self.conv.in_channels, \
            f"聚合层输入通道错误：实际{x.shape[-1]}，期望{self.conv.in_channels}"

        x = x.permute(0, 3, 1, 2)  # (B, C, M, K)
        attn_weights = self.attn(x)  # (B, out_C, M, K)
        x = self.conv(x) * attn_weights  # (B, out_C, M, K)
        x = self.bn(x)
        x = F.relu(x)
        x = x.mean(dim=-1)  # (B, out_C, M)
        return x.permute(0, 2, 1)  # (B, M, out_C)


class CSIT(nn.Module):
    """跨尺度交互Transformer（适配64通道输入，增强稳定性）"""

    def __init__(self, dim=64):  # 与输入特征通道匹配（当前配置为64）
        super().__init__()
        self.dim = dim
        self.num_heads = 4  # 64通道适配4头注意力（64/4=16，更合理的头维度）
        self.scale = (dim // self.num_heads) ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim)  # 输入通道匹配dim
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_low, x_high):
        # 修正：通道数检查（应为最后一个维度）
        assert x_low.shape[-1] == x_high.shape[-1] == self.dim, \
            f"CSIT输入通道不匹配：x_low={x_low.shape[-1]}, x_high={x_high.shape[-1]}, 期望{self.dim}"

        B, N, _ = x_low.shape
        M = x_high.shape[1]
        residual = x_low  # 残差连接

        # 拼接特征（通道均为dim）
        x = torch.cat([x_low, x_high], dim=1)  # (B, N+M, dim)
        qkv = self.qkv_proj(x).reshape(B, N + M, 3, self.num_heads, self.dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 注意力计算（添加数值稳定性处理）
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 防止softmax数值溢出
        attn = attn - attn.max(dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        # 添加注意力 dropout 增强泛化性
        attn = F.dropout(attn, p=0.1, training=self.training)

        x_interact = (attn @ v).transpose(1, 2).reshape(B, N + M, self.dim)
        x_interact = self.out_proj(x_interact)

        # 提取低尺度特征并添加残差
        x_low_interact = x_interact[:, :N, :]
        return self.norm(x_low_interact + residual)