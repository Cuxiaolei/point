# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 工具函数：最远点采样（GPU 上运行）
# -------------------------
@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, m: int):
    """
    批量最远点采样 (B, N, 3) -> (B, m)
    说明：
      - 使用纯 PyTorch 张量运算；当 xyz 在 cuda 上时，计算在 GPU 上进行
      - 与常见的 CUDA Kernel FPS 相比略慢，但无需编译，且对 N=8k, m<=512/1024 可接受
    """
    assert xyz.ndim == 3 and xyz.size(-1) == 3, "fps 输入应为 (B, N, 3)"
    device = xyz.device
    B, N, _ = xyz.shape
    m = min(m, N)

    centroids = torch.zeros(B, m, dtype=torch.long, device=device)
    # 初始化为 +inf
    distances = torch.full((B, N), float("inf"), device=device)

    # 随机选择一个起始点（你也可以选择几何中心最近点等别的策略）
    farthest = torch.randint(0, N, (B,), device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(m):
        centroids[:, i] = farthest
        # 当前最远点的坐标
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B,1,3)
        # 计算到所有点的欧氏距离的平方
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # (B,N)
        # 维护最近距离
        mask = dist < distances
        distances[mask] = dist[mask]
        # 选择全局最远的最近距离点
        farthest = torch.max(distances, dim=1).indices

    return centroids  # (B, m)


# -------------------------
# 工具函数：批量 kNN（GPU 上运行）
# -------------------------
@torch.no_grad()
def knn_indices(query_xyz: torch.Tensor, ref_xyz: torch.Tensor, k: int):
    """
    在 ref_xyz 中检索 query_xyz 的 kNN
    query_xyz: (B, Q, 3)
    ref_xyz:   (B, N, 3)
    返回 idx:   (B, Q, k)
    """
    B, Q, _ = query_xyz.shape
    _, N, _ = ref_xyz.shape
    k = min(k, N)
    # pairwise 距离 (B, Q, N)
    # 注意：torch.cdist 在 GPU 上也很快
    dists = torch.cdist(query_xyz, ref_xyz, p=2)  # (B,Q,N)
    # 取最小的 k 个
    idx = torch.topk(dists, k=k, dim=-1, largest=False).indices  # (B,Q,k)
    return idx


def batched_index_points(points: torch.Tensor, idx: torch.Tensor):
    """
    根据批量索引提取点/特征
    points: (B, N, C)
    idx:    (B, M, k) 或 (B, M)
    return: (B, M, k, C) 或 (B, M, C)
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1)
    if idx.ndim == 3:
        B, M, K = idx.shape
        batch_indices = batch_indices.expand(-1, M, K)
        return points[batch_indices, idx, :]
    else:
        B, M = idx.shape
        batch_indices = batch_indices.expand(-1, M)
        return points[batch_indices, idx, :]


# -------------------------
# 小模块：简单的 EdgeConv / 局部聚合
# -------------------------
class LocalAgg(nn.Module):
    def __init__(self, in_ch, out_ch, k=16):
        super().__init__()
        self.k = k
        # EdgeConv 风格：phi([x_i, x_j - x_i])
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor):
        """
        feats: (B, N, C)
        xyz:   (B, N, 3)
        返回:  (B, N, out_ch)
        """
        B, N, C = feats.shape
        k = min(self.k, N)
        idx = knn_indices(xyz, xyz, k)  # (B,N,k)

        # 中心特征与邻域特征
        neighbor = batched_index_points(feats, idx)          # (B,N,k,C)
        center = feats.unsqueeze(2).expand(-1, -1, k, -1)    # (B,N,k,C)
        edge = torch.cat([center, neighbor - center], dim=-1)  # (B,N,k,2C)

        # 转为 conv2d 需要的 [B, C, N, k]
        edge = edge.permute(0, 3, 1, 2).contiguous()  # (B,2C,N,k)
        out = self.mlp(edge)                          # (B,out_ch,N,k)
        out = torch.max(out, dim=-1).values           # (B,out_ch,N)
        out = out.permute(0, 2, 1).contiguous()       # (B,N,out_ch)
        return out


# -------------------------
# 上采样（基于 最近邻/三邻居插值）
# -------------------------
def three_nn_interpolate(xyz_src, xyz_dst, feat_src):
    """
    把 src 的特征插值到 dst（3-NN 权重插值）
    xyz_src:  (B, N1, 3)
    xyz_dst:  (B, N2, 3)
    feat_src: (B, N1, C)
    返回:     (B, N2, C)
    """
    B, N2, _ = xyz_dst.shape
    # 找到 dst 在 src 的 3 个近邻
    idx = knn_indices(xyz_dst, xyz_src, k=3)  # (B,N2,3)
    neighbor = batched_index_points(xyz_src, idx)  # (B,N2,3,3)
    dist = torch.norm(neighbor - xyz_dst.unsqueeze(2), dim=-1) + 1e-8  # (B,N2,3)

    inv = 1.0 / dist
    weights = inv / torch.sum(inv, dim=-1, keepdim=True)  # 归一化权重 (B,N2,3)

    src_feat_knn = batched_index_points(feat_src, idx)  # (B,N2,3,C)
    out = torch.sum(src_feat_knn * weights.unsqueeze(-1), dim=2)  # (B,N2,C)
    return out


# -------------------------
# 主模型：SGDAT（简洁稳定版）
# -------------------------
class SGDAT(nn.Module):
    def __init__(self, num_classes: int, base_dim: int = 64,
                 max_points: int = 8000, k: int = 16, debug: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.base = base_dim
        self.max_points = max_points
        self.k = k
        self.debug = debug

        # -------- 输入处理：坐标归一化 + 线性投影  --------
        # 输入特征：9 维（xyz + rgb + normal），但也兼容 C!=9（会自动适配）
        # 这里先用一个 1x1 卷积把输入映射到 base 维
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=self.base, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.base),
            nn.ReLU(inplace=True)
        )

        # -------- 层次编码：N -> 512 -> 128 --------
        self.local1 = LocalAgg(in_ch=self.base, out_ch=self.base, k=self.k)          # N
        self.local2 = LocalAgg(in_ch=self.base, out_ch=self.base * 2, k=self.k)      # 512
        self.local3 = LocalAgg(in_ch=self.base * 2, out_ch=self.base * 2, k=self.k)  # 128

        # 下采样后的特征压缩/融合
        self.down1 = nn.Sequential(
            nn.Conv1d(self.base * 2, self.base, 1, bias=False),
            nn.BatchNorm1d(self.base),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(self.base * 2, self.base * 2, 1, bias=False),
            nn.BatchNorm1d(self.base * 2),
            nn.ReLU(inplace=True)
        )

        # 上采样融合
        self.up1 = nn.Sequential(  # 128 -> 512
            nn.Conv1d(self.base * 3, self.base * 2, 1, bias=False),
            nn.BatchNorm1d(self.base * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(  # 512 -> N
            nn.Conv1d(self.base * 3, self.base, 1, bias=False),
            nn.BatchNorm1d(self.base),
            nn.ReLU(inplace=True),
        )

        # 语义分割头（每点分类）
        self.head = nn.Sequential(
            nn.Conv1d(self.base, self.base, 1, bias=False),
            nn.BatchNorm1d(self.base),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv1d(self.base, num_classes, 1, bias=True),
        )

        # 供 Trainer 注入的类别权重
        self.class_weights = None

    def _split_xyz_feat(self, x: torch.Tensor):
        """
        x: (B, N, C) 期望 C>=3; 前 3 维是 xyz，其余视为特征
        如果 C<9 也能工作，但推荐使用 9 维 (xyz+rgb+normal)
        """
        assert x.ndim == 3 and x.size(-1) >= 3, "输入特征维度至少包含 xyz"
        xyz = x[..., :3]
        feat = x[..., 3:]
        return xyz, feat

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C)  — C 默认 9（xyz+rgb+normal）
        return logits: (B, N, num_classes)
        """
        if self.debug and not hasattr(self, "_printed_shape"):
            print(f"[DEBUG] Input shape: {x.shape}")
            self._printed_shape = True

        B, N, C = x.shape
        device = x.device

        # 拆分坐标 / 特征，并对坐标做简单归一化（中心化 + 尺度归一）
        xyz, attr = self._split_xyz_feat(x)  # (B,N,3), (B,N,C-3)

        xyz_mean = xyz.mean(dim=1, keepdim=True)
        xyz_centered = xyz - xyz_mean
        # 尺度：使用标准差（防止归一化过度）
        xyz_std = torch.clamp(xyz_centered.std(dim=1, keepdim=True), min=1e-3)
        xyz_normed = xyz_centered / xyz_std  # (B,N,3)

        # 准备 9 维输入：若 C==9 则直接用；否则拼接（xyz_normed + 原始 attr），并在不足 9 时补零
        if C >= 9:
            x9 = torch.cat([xyz_normed, x[..., 3:6], x[..., 6:9]], dim=-1)  # 安全地取到 9 维
            x9 = x9[:, :, :9]  # 防止 C>9 时溢出
        else:
            # 不足 9 维时用 0 填充（仍然可跑，但建议你提供 9 维）
            pad = torch.zeros(B, N, max(0, 9 - C), device=device, dtype=x.dtype)
            x9 = torch.cat([x, pad], dim=-1)

        # 1x1 proj 到 base
        f = self.in_proj(x9.permute(0, 2, 1))  # (B,base,N)
        f = f.permute(0, 2, 1).contiguous()    # (B,N,base)

        # -------- 层次编码 + FPS 下采样 --------
        # L0: N
        f0 = self.local1(f, xyz_normed)  # (B,N,base)

        # 采样到 512
        m1 = min(512, N)
        idx_512 = farthest_point_sample(xyz_normed, m1)  # (B,512)
        xyz_512 = batched_index_points(xyz_normed, idx_512)  # (B,512,3)
        f_512 = batched_index_points(f0, idx_512)            # (B,512,base)

        f1 = self.local2(f_512, xyz_512)  # (B,512,2*base)
        f1_red = self.down1(f1.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,base)

        # 采样到 128
        m2 = min(128, m1)
        idx_128 = farthest_point_sample(xyz_512, m2)  # (B,128)
        xyz_128 = batched_index_points(xyz_512, idx_128)  # (B,128,3)
        f_128 = batched_index_points(f1, idx_128)         # (B,128,2*base)

        f2 = self.local3(f_128, xyz_128)  # (B,128,2*base)
        f2_red = self.down2(f2.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,128,2*base)

        if self.debug:
            with torch.no_grad():
                n_nan = torch.isnan(f1_red).any().item() or torch.isnan(f2_red).any().item()
                n_inf = torch.isinf(f1_red).any().item() or torch.isinf(f2_red).any().item()
                print(f"[DEBUG] out1 shape: {f1_red.shape}, NaN={int(n_nan)}, Inf={int(n_inf)}")
                print(f"[DEBUG] out2 shape: {f2_red.shape}, NaN={int(n_nan)}, Inf={int(n_inf)}")

        # -------- 上采样到 512，再到 N --------
        up_512 = three_nn_interpolate(xyz_128, xyz_512, f2_red)   # (B,512,2*base)
        fuse_512 = torch.cat([f1, f1_red, up_512], dim=-1)        # (B,512, 2b + b + 2b = 5b)
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,2b)

        up_N = three_nn_interpolate(xyz_512, xyz_normed, fuse_512)  # (B,N,2b)
        fuse_N = torch.cat([f0, f, up_N], dim=-1)                   # (B,N, b + b + 2b = 4b)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,b)

        # -------- 每点分类头 --------
        logits = self.head(fuse_N.permute(0, 2, 1))  # (B,num_classes,N)
        logits = logits.permute(0, 2, 1).contiguous()  # (B,N,num_classes)

        if self.debug and not hasattr(self, "_printed_head"):
            print(f"[DEBUG] sem1_logits shape: {logits.shape}, NaN={int(torch.isnan(logits).any())}, Inf={int(torch.isinf(logits).any())}")
            self._printed_head = True

        return logits  # (B, N, num_classes)

    def get_loss(self, logits, labels, class_weights=None):
        """
        logits: [B, N, num_classes]
        labels: [B, N]
        """
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float()
        acc = acc[labels != -1].mean()  # 忽略 padding
        return loss, preds, {"acc": acc.item()}

