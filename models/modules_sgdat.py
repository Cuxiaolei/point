# modules_sgdat.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 初始化工具
# ------------------------------------------------------------
def init_linear(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_conv1x1(m: nn.Module):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ------------------------------------------------------------
# 基础模块
# ------------------------------------------------------------
class MLP1d(nn.Module):
    """
    1x1 Conv -> BN -> ReLU 的堆叠。输入/输出: [B, C_in, N] -> [B, C_out, N]
    """
    def __init__(self, c_in: int, c_out: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=1, bias=not use_bn)
        self.bn = nn.BatchNorm1d(c_out) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 1e-6 else nn.Identity()
        self.apply(init_conv1x1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, N]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class SE1d(nn.Module):
    """
    Squeeze-Excitation for 1D features. 输入 [B, C, N]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        y = self.pool(x).view(b, c)               # [B, C]
        y = self.fc2(self.act(self.fc1(y)))       # [B, C]
        y = self.gate(y).view(b, c, 1)            # [B, C, 1]
        return x * y


class Residual(nn.Module):
    """
    残差封装：y = x + F(x)（若维度不匹配则使用1x1投影）
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.need_proj = (c_in != c_out)
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False) if self.need_proj else nn.Identity()
        if self.need_proj:
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        if self.need_proj:
            x = self.proj(x)
        return x + fx


# ------------------------------------------------------------
# 邻域检索与采样
# ------------------------------------------------------------
def knn_indices(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    简单 KNN（基于欧氏距离的全连接计算）:
    coords: [B, 3, N]
    return: idx [B, N, k]
    """
    b, _, n = coords.shape
    # [B, N, N]
    with torch.no_grad():
        dist2 = torch.cdist(coords.transpose(1, 2).contiguous(), coords.transpose(1, 2).contiguous(), p=2)
        # 取最近邻（包含自身）
        idx = dist2.topk(k=k, largest=False)[1]   # [B, N, k]
    return idx


def gather_neighbor(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C, N]
    idx:  [B, N, k]
    return: [B, C, N, k]
    """
    b, c, n = feat.shape
    k = idx.shape[-1]
    idx_expand = idx.unsqueeze(1).expand(b, c, n, k)  # [B, C, N, K]
    feat_expand = feat.unsqueeze(2).expand(b, c, n, n)
    # 使用 take_along_dim 选取邻域特征
    neigh = torch.take_along_dim(feat_expand, idx_expand, dim=3)  # [B, C, N, K]
    return neigh


def hybrid_fps_random(xyz: torch.Tensor, m: int, fps_ratio: float = 0.7) -> torch.Tensor:
    """
    混合采样：FPS + 随机
    xyz: [B, 3, N]
    return: idx [B, m]
    """
    b, _, n = xyz.shape
    device = xyz.device
    m = min(m, n)
    fps_m = int(m * fps_ratio)
    rand_m = m - fps_m

    # --- FPS 简化实现（朴素 O(B*N*m)），在 N 较大时可用更高效实现 ---
    with torch.no_grad():
        idxs = []
        for bi in range(b):
            pts = xyz[bi].transpose(0, 1).contiguous()  # [N, 3]
            # 随机初始化一个中心
            farthest = torch.randint(0, n, (1,), device=device)
            centroids = [farthest.item()]
            dist = torch.full((n,), 1e10, device=device)
            for _ in range(max(1, fps_m) - 1):
                centroid = pts[centroids[-1]].unsqueeze(0)  # [1, 3]
                d = torch.sum((pts - centroid) ** 2, dim=1)
                dist = torch.minimum(dist, d)
                farthest = torch.argmax(dist)
                centroids.append(int(farthest))
            centroids = torch.tensor(centroids, device=device, dtype=torch.long)
            if rand_m > 0:
                rand_idx = torch.randperm(n, device=device)[:rand_m]
                out = torch.cat([centroids, rand_idx], dim=0)
            else:
                out = centroids
            # 去重 & 截断
            out = torch.unique_consecutive(out)[:m]
            if out.numel() < m:
                # 不足则补随机
                need = m - out.numel()
                extra = torch.randperm(n, device=device)[:need]
                out = torch.cat([out, extra], dim=0)
            idxs.append(out.unsqueeze(0))
        idx = torch.cat(idxs, dim=0)  # [B, m]
    return idx


# ------------------------------------------------------------
# 动态半径通道融合（改进版）
# ------------------------------------------------------------
class DynamicRadiusChannelFusion(nn.Module):
    """
    输入:
        x:     [B, C, N]   特征
        pos:   [B, 3, N]   坐标
    过程:
        - KNN 邻域
        - 距离 → 权重 softmax(-d/τ)
        - 距离加权聚合（减少过度平滑）
        - 通道 MLP + BN + ReLU + SE + 残差
    输出:
        y:     [B, C_out, N]
    """
    def __init__(
        self,
        c_in: int,
        c_hidden: Optional[int] = None,
        c_out: Optional[int] = None,
        k: int = 16,
        tau: float = 0.2,
        reduction: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        c_hidden = c_hidden or c_in
        c_out = c_out or c_in
        self.k = k
        self.tau = tau

        # 邻域增强后的通道融合
        self.channel_proj = nn.Sequential(
            MLP1d(c_in * 2, c_hidden, use_bn=True, dropout=dropout),
            MLP1d(c_hidden, c_out, use_bn=True, dropout=dropout),
        )
        self.se = SE1d(c_out, reduction=reduction)
        self.res = Residual(c_in, c_out)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # 保护数值
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)

        b, c, n = x.shape
        # KNN
        idx = knn_indices(pos, k=self.k)  # [B, N, k]
        # 邻域特征
        neigh_x = gather_neighbor(x, idx)  # [B, C, N, k]
        # 邻域坐标用于计算距离
        pos_expand = pos.unsqueeze(-1).expand(b, 3, n, self.k)             # [B, 3, N, k]
        neigh_pos = gather_neighbor(pos, idx)                               # [B, 3, N, k]
        d = torch.norm(neigh_pos - pos_expand, dim=1)                       # [B, N, k]
        d = torch.clamp(d, min=1e-6)
        # 权重：softmax(-d/τ)
        w = F.softmax(-d / max(1e-6, self.tau), dim=-1).unsqueeze(1)        # [B, 1, N, k]
        # 加权聚合
        agg = torch.sum(neigh_x * w, dim=-1)                                # [B, C, N]

        # 拼接 (x || agg) 进行通道融合
        fuse = torch.cat([x, agg], dim=1)                                   # [B, 2C, N]
        y = self.channel_proj(fuse)                                         # [B, C_out, N]
        y = self.se(y)
        y = self.res(x, y)                                                  # 残差
        return y


# ------------------------------------------------------------
# 轻量多头注意力（全局）
# ------------------------------------------------------------
class MultiHeadSelfAttention1D(nn.Module):
    """
    多头自注意力（简化版），输入为 [B, C, N]，输出同形状。
    默认使用缩放点积注意力；为了鲁棒性，对 logits/attn 做了 clamp。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 1e-6 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 1e-6 else nn.Identity()

        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N] -> 转为 [B, N, C]
        x_ = x.transpose(1, 2).contiguous()         # [B, N, C]
        b, n, c = x_.shape

        qkv = self.qkv(x_)                          # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)              # [B, N, C] * 3

        # 分头
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]

        # 缩放点积注意力
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        attn_logits = torch.nan_to_num(attn_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                       # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(b, n, c)              # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)

        # 回到 [B, C, N]
        out = out.transpose(1, 2).contiguous()
        return out


class LinearSpatialGVA(nn.Module):
    """
    兼容原名的全局注意力模块（实现为多头注意力 + 前馈 MLP + 残差）
    输入/输出: [B, C, N]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.attn = MultiHeadSelfAttention1D(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.attn_res = Residual(dim, dim)

        self.ffn = nn.Sequential(
            MLP1d(dim, hidden, use_bn=True, dropout=dropout),
            MLP1d(hidden, dim, use_bn=True, dropout=dropout),
        )
        self.ffn_res = Residual(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力残差
        y = self.attn(x)
        x = self.attn_res(x, y)
        # FFN 残差
        y = self.ffn(x)
        x = self.ffn_res(x, y)
        return x


# ------------------------------------------------------------
# 金字塔/层级块（示例）：局部聚合 + 全局注意力
# ------------------------------------------------------------
class LocalGlobalFusionBlock(nn.Module):
    """
    一个典型层：局部动态半径通道融合 + 全局注意力
    输入/输出: [B, C, N]
    """
    def __init__(
        self,
        c_in: int,
        c_mid: Optional[int] = None,
        c_out: Optional[int] = None,
        k: int = 16,
        tau: float = 0.2,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        c_mid = c_mid or c_in
        c_out = c_out or c_in

        self.local = DynamicRadiusChannelFusion(
            c_in=c_in, c_hidden=c_mid, c_out=c_out, k=k, tau=tau, dropout=dropout
        )
        self.global_attn = LinearSpatialGVA(
            dim=c_out, num_heads=heads, mlp_ratio=2.0, attn_drop=0.0, proj_drop=dropout, dropout=dropout
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.local(x, pos)            # [B, C_out, N]
        x = self.global_attn(x)           # [B, C_out, N]
        return x


# ------------------------------------------------------------
# 下采样封装（可供主干使用）
# ------------------------------------------------------------
class DownsampleWithHybridSampler(nn.Module):
    """
    通过混合采样（FPS+随机）下采样点集，并用 KNN 从原集聚合到子集特征。
    输入:
        x:   [B, C, N]
        pos: [B, 3, N]
        m:   目标点数
    输出:
        x_sub:   [B, C, m]
        pos_sub: [B, 3, m]
    """
    def __init__(self, k: int = 16, fps_ratio: float = 0.7, tau: float = 0.2):
        super().__init__()
        self.k = k
        self.fps_ratio = fps_ratio
        self.tau = tau

    def forward(self, x: torch.Tensor, pos: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, n = x.shape
        m = min(m, n)
        idx = hybrid_fps_random(pos, m=m, fps_ratio=self.fps_ratio)     # [B, m]
        # 取子集坐标
        pos_sub = torch.gather(pos, dim=2, index=idx.unsqueeze(1).expand(b, 3, m))  # [B, 3, m]

        # 用子集坐标对原集 KNN，把原特征聚合到子集
        with torch.no_grad():
            # [B, m, N]
            dist2 = torch.cdist(pos_sub.transpose(1, 2).contiguous(), pos.transpose(1, 2).contiguous(), p=2)
            knn_idx = dist2.topk(k=self.k, largest=False)[1]                    # [B, m, k]

        neigh_x = gather_neighbor(x, knn_idx)                                    # [B, C, m, k]
        neigh_pos = gather_neighbor(pos, knn_idx)                                 # [B, 3, m, k]
        pos_sub_expand = pos_sub.unsqueeze(-1).expand(b, 3, m, self.k)
        d = torch.norm(neigh_pos - pos_sub_expand, dim=1)                        # [B, m, k]
        d = torch.clamp(d, min=1e-6)
        w = F.softmax(-d / max(1e-6, self.tau), dim=-1).unsqueeze(1)             # [B, 1, m, k]
        x_sub = torch.sum(neigh_x * w, dim=-1)                                   # [B, C, m]
        return x_sub, pos_sub
