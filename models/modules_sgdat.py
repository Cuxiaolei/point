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
    with torch.no_grad():
        dist2 = torch.cdist(coords.transpose(1, 2).contiguous(), coords.transpose(1, 2).contiguous(), p=2)
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

    with torch.no_grad():
        idxs = []
        for bi in range(b):
            pts = xyz[bi].transpose(0, 1).contiguous()  # [N, 3]
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
            out = torch.unique_consecutive(out)[:m]
            if out.numel() < m:
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

        self.channel_proj = nn.Sequential(
            MLP1d(c_in * 2, c_hidden, use_bn=True, dropout=dropout),
            MLP1d(c_hidden, c_out, use_bn=True, dropout=dropout),
        )
        self.se = SE1d(c_out, reduction=reduction)
        self.res = Residual(c_in, c_out)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)

        b, c, n = x.shape
        idx = knn_indices(pos, k=self.k)  # [B, N, k]
        neigh_x = gather_neighbor(x, idx)  # [B, C, N, k]

        pos_expand = pos.unsqueeze(-1).expand(b, 3, n, self.k)             # [B, 3, N, k]
        neigh_pos = gather_neighbor(pos, idx)                               # [B, 3, N, k]
        d = torch.norm(neigh_pos - pos_expand, dim=1)                       # [B, N, k]
        d = torch.clamp(d, min=1e-6)
        w = F.softmax(-d / max(1e-6, self.tau), dim=-1).unsqueeze(1)        # [B, 1, N, k]
        agg = torch.sum(neigh_x * w, dim=-1)                                # [B, C, N]

        fuse = torch.cat([x, agg], dim=1)                                   # [B, 2C, N]
        y = self.channel_proj(fuse)                                         # [B, C_out, N]
        y = self.se(y)
        y = self.res(x, y)
        return y


# ------------------------------------------------------------
# 轻量多头注意力（原版，全局）
# ------------------------------------------------------------
class MultiHeadSelfAttention1D(nn.Module):
    """
    多头自注意力（简化版），输入为 [B, C, N]，输出同形状。
    注意：该实现为标准 Softmax(QK^T) 的 O(N^2) 复杂度，仅保留以兼容历史。
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
        # x: [B, C, N] -> [B, N, C]
        x_ = x.transpose(1, 2).contiguous()
        b, n, c = x_.shape

        qkv = self.qkv(x_)                          # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)              # [B, N, C] * 3

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        attn_logits = torch.nan_to_num(attn_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                       # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(b, n, c)              # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.transpose(1, 2).contiguous()                            # [B, C, N]
        return out


# ------------------------------------------------------------
# 线性多头注意力（O(N) 复杂度，全局）
# ------------------------------------------------------------
class LinearMultiHeadSelfAttention1D(nn.Module):
    """
    线性复杂度多头注意力（Performer 风格非负核 trick）：
        phi(x) = elu(x) + 1  （逐元素正映射）
        out_i = (phi(q_i) @ (K^T V)) / (phi(q_i) @ sum_j phi(k_j))

    输入/输出: [B, C, N] -> [B, C, N]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,  # 占位，接口兼容
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 1e-6 else nn.Identity()

        self.apply(init_linear)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        # 非负特征映射，避免 softmax 显存开销；保持数值稳定
        return F.elu(x, inplace=False) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N] -> [B, N, C]
        x_ = x.transpose(1, 2).contiguous()
        b, n, c = x_.shape

        qkv = self.qkv(x_)                          # [B, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)              # [B, N, C] * 3

        # [B, H, N, Dh]
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 线性核映射
        q_phi = self._phi(q)                         # [B, H, N, Dh]
        k_phi = self._phi(k)                         # [B, H, N, Dh]

        # 预计算项：
        # S = K^T V  -> [B, H, Dh, Dh]
        # k_sum = sum_j K_j -> [B, H, Dh]
        # 注意：全部是 O(N * Dh^2) / O(N * Dh)
        S = torch.matmul(k_phi.transpose(-2, -1), v)            # [B, H, Dh, Dh]
        k_sum = k_phi.sum(dim=-2)                                # [B, H, Dh]

        # 分子：Q * S   -> [B, H, N, Dh]
        numerator = torch.matmul(q_phi, S)                       # [B, H, N, Dh]
        # 分母：<Q, sum K>  -> [B, H, N]
        denominator = torch.einsum('bhnd,bhd->bhn', q_phi, k_sum).unsqueeze(-1)  # [B, H, N, 1]
        denominator = denominator + self.eps

        out = numerator / denominator                            # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(b, n, c)     # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).contiguous()                   # [B, C, N]
        return out


class LinearSpatialGVA(nn.Module):
    """
    兼容原名的全局注意力模块（改为线性注意力 + 前馈 MLP + 残差）
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

        # 将注意力替换为 O(N) 的线性版本，接口保持一致
        self.attn = LinearMultiHeadSelfAttention1D(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
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
        y = self.attn(x)
        x = self.attn_res(x, y)
        y = self.ffn(x)
        x = self.ffn_res(x, y)
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
        pos_sub = torch.gather(pos, dim=2, index=idx.unsqueeze(1).expand(b, 3, m))  # [B, 3, m]

        with torch.no_grad():
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


# ============================================================
# DropPath
# ============================================================
class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath: 以概率 p 将输入乘以 0，并按(1-p)进行期望缩放。
    输入输出同形状，通常包裹在残差分支上。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# ============================================================
# ChannelCCC（通道相关性注意力）
# ============================================================
class ChannelCCC(nn.Module):
    """
    基于通道相关性（相似度矩阵）的轻量注意力。
    输入/输出: [B, C, N]
    """
    def __init__(self, channels: int, proj_ratio: float = 1.0, dropout: float = 0.0):
        super().__init__()
        c_proj = int(round(channels * proj_ratio))
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_out.weight, mode="fan_out", nonlinearity="relu")
        self.drop = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        xm = x - x.mean(dim=-1, keepdim=True)
        xn = F.normalize(xm, p=2, dim=-1)
        G = torch.matmul(xn, xn.transpose(1, 2)) / max(1, n)  # [B, C, C]
        G = torch.nan_to_num(G)
        A = F.softmax(G, dim=-1)
        y = torch.matmul(A, x)                                # [B, C, N]
        y = self.proj_out(y)
        y = self.drop(y)
        return x + self.gamma * y


# ============================================================
# 跨尺度最近邻插值（channel-first）
# ============================================================
def nearest_interpolate(target_pos: torch.Tensor,
                        source_pos: torch.Tensor,
                        source_feat: torch.Tensor) -> torch.Tensor:
    """
    将 source_feat 从 source_pos 最近邻插值到 target_pos。
    target_pos: [B, 3, Nt]
    source_pos: [B, 3, Ns]
    source_feat: [B, C, Ns]
    return: [B, C, Nt]
    """
    b, _, nt = target_pos.shape
    _, c, ns = source_feat.shape
    with torch.no_grad():
        dist = torch.cdist(target_pos.transpose(1, 2).contiguous(),
                           source_pos.transpose(1, 2).contiguous(), p=2)  # [B, Nt, Ns]
        idx = dist.argmin(dim=-1)  # [B, Nt]
    idx_expand = idx.unsqueeze(1).expand(b, c, nt)  # [B, C, Nt]
    out = torch.gather(source_feat, dim=2, index=idx_expand)
    return out


# ============================================================
# 语义引导门控（SGF 的门控单元）
# ============================================================
class SemanticGuidedGate(nn.Module):
    """
    用粗尺度语义先验（logits 或 prob）对上采样分支进行空间门控。
    典型用法：
        # 128 尺度得到的 logits: sem128_logits [B, K, 128]
        # 需要对 512 尺度的上采样特征进行门控：
        gate_512 = sg_gate(sem_logits=sem128_logits,
                           source_pos=xyz_128, target_pos=xyz_512)  # [B, 1, 512]
        up128_to_512 = up128_to_512 * gate_512
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Conv1d(num_classes, 1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.gate = nn.Sigmoid()

    @torch.no_grad()
    def _to_prob(self, sem_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(sem_logits, dim=1)

    def forward(self,
                sem_logits: torch.Tensor,
                source_pos: torch.Tensor,
                target_pos: torch.Tensor) -> torch.Tensor:
        """
        sem_logits: [B, K, Ms]   （粗尺度）
        source_pos: [B, 3, Ms]
        target_pos: [B, 3, Nt]
        return: gate [B, 1, Nt]
        """
        prob = self._to_prob(sem_logits)                 # [B, K, Ms]
        prob_up = nearest_interpolate(target_pos, source_pos, prob)  # [B, K, Nt]
        gate = self.gate(self.conv(prob_up))             # [B, 1, Nt]
        return gate
