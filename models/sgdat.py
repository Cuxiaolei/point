import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 工具 ---------
def farthest_point_sample(x, m):  # x:(B,N,3) -> (B,m)
    # 朴素版 FPS，B 小时够用；若想提速可换 CUDA 实现
    B, N, _ = x.shape
    centroids = torch.zeros(B, m, dtype=torch.long, device=x.device)
    distance = torch.full((B, N), 1e10, device=x.device)
    farthest = torch.randint(0, N, (B,), device=x.device)
    batch_indices = torch.arange(B, device=x.device)
    for i in range(m):
        centroids[:, i] = farthest
        centroid = x[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids  # (B,m)

def knn(center, points, k):  # center:(B,M,3) points:(B,N,3) -> idx:(B,M,k)
    # 基于欧式距离的 KNN
    with torch.no_grad():
        dist = torch.cdist(center, points)  # (B,M,N)
        idx = torch.topk(dist, k=k, dim=-1, largest=False)[1]  # (B,M,k)
    return idx

def index_points(points, idx):  # points:(B,N,C), idx:(B,M) or (B,M,k)
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(*idx.shape[:2], 1)
    if idx.dim() == 2:
        return points[torch.arange(B).unsqueeze(-1), idx]  # (B,M,C)
    else:
        batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(B, idx.shape[1], idx.shape[2])
        return points[batch_indices, idx]  # (B,M,k,C)

# --------- 层 ---------
class MLP1d(nn.Module):
    def __init__(self, in_c, out_c, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.SiLU(inplace=True),
            nn.LayerNorm(out_c),
            nn.Dropout(p),
            nn.Linear(out_c, out_c),
            nn.SiLU(inplace=True),
            nn.LayerNorm(out_c),
            nn.Dropout(p),
        )
    def forward(self, x):  # (B,*,C)
        return self.net(x)

class NeighborhoodFuse(nn.Module):
    """
    将中心点与其 kNN 邻域做特征融合：
      输入: centers (B,M,3), feats (B,N,F), points (B,N,3), knn_idx (B,M,k)
      融合: [center_feat, neighbor_feat-center_feat, pos_delta]
      输出: (B,M,out_dim)
    """
    def __init__(self, in_feat, out_feat, k=16, p=0.0):
        super().__init__()
        self.k = k
        self.mlp = MLP1d(in_feat + in_feat + 3, out_feat, p=p)
        self.proj = nn.Linear(out_feat, out_feat)

    def forward(self, centers_xyz, feats, points_xyz, knn_idx):
        # feats:(B,N,F), points_xyz:(B,N,3)
        B, M, _ = centers_xyz.shape
        k = self.k
        # 取邻域
        neigh_xyz = index_points(points_xyz, knn_idx)           # (B,M,k,3)
        neigh_feat = index_points(feats, knn_idx)               # (B,M,k,F)
        # 中心对应的特征
        # 先找每个 centers_xyz 在 points_xyz 中的最近点索引用于取中心特征
        with torch.no_grad():
            nn_idx = torch.cdist(centers_xyz, points_xyz).argmin(dim=-1)  # (B,M)
        center_feat = index_points(feats, nn_idx)               # (B,M,F)
        center_feat_exp = center_feat.unsqueeze(2).expand(B, M, k, center_feat.shape[-1])

        pos_delta = neigh_xyz - centers_xyz.unsqueeze(2)        # (B,M,k,3)
        fuse = torch.cat([center_feat_exp, neigh_feat - center_feat_exp, pos_delta], dim=-1)  # (B,M,k, 2F+3)
        fuse = self.mlp(fuse)                                   # (B,M,k,out)
        fuse = fuse.mean(dim=2)                                 # (B,M,out)
        return self.proj(fuse)                                  # (B,M,out)

# --------- 主模型 ---------
class SGDAT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 k: int = 16,
                 m1: int = 512,
                 m2: int = 128,
                 base_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.m1 = m1
        self.m2 = m2

        # 初始点特征（只用坐标）
        self.stem = nn.Sequential(
            nn.Linear(3, base_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(base_dim)
        )

        # 两级融合
        self.fuse1 = NeighborhoodFuse(base_dim, base_dim, k=k, p=dropout)  # (B,m1,base_dim)
        self.fuse2 = NeighborhoodFuse(base_dim, base_dim, k=k, p=dropout)  # (B,m2,base_dim)

        # 语义头（一级辅助 + 全局主头）
        self.sem1_head = nn.Linear(base_dim, num_classes)   # (B,m1,C)
        self.sem2_head = nn.Linear(base_dim, base_dim)      # 供上采样融合
        self.main_head = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(base_dim),
            nn.Dropout(dropout),
            nn.Linear(base_dim, num_classes)
        )

        # 上采样时与原点拼接的 stem
        self.back_proj = nn.Sequential(
            nn.Linear(base_dim + base_dim, base_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(base_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    @torch.no_grad()
    def _nearest_index(self, src_xyz, dst_xyz):
        # 对 dst 中每个点，在 src 中找最近的点索引
        # src:(B,Ms,3) dst:(B,N,3) -> idx:(B,N)
        dist = torch.cdist(dst_xyz, src_xyz)   # (B,N,Ms)
        idx = dist.argmin(dim=-1)              # (B,N)
        return idx

    def forward(self, points):
        """
        points: (B,N,3)
        return:
          logits: (B,N,C)
          aux: dict(sem1_logits:(B,m1,C), idx1:(B,m1), idx2:(B,m2))
        """
        B, N, _ = points.shape
        xyz = points

        # stem
        feats = self.stem(xyz)  # (B,N,F)

        # FPS
        idx1 = farthest_point_sample(xyz, self.m1)  # (B,m1)
        centers1 = index_points(xyz, idx1)          # (B,m1,3)
        knn1 = knn(centers1, xyz, self.k)           # (B,m1,k)
        out1 = self.fuse1(centers1, feats, xyz, knn1)  # (B,m1,F)
        sem1_logits = self.sem1_head(out1)          # (B,m1,C)

        # 第二级：在 out1 的中心再做 FPS
        with torch.no_grad():
            # 在 centers1 上再采样 m2
            idx2_local = farthest_point_sample(centers1, self.m2)   # (B,m2) in local m1 index
            idx2 = torch.gather(idx1, 1, idx2_local)                # (B,m2) map 回原 N 空间索引
        centers2 = index_points(xyz, idx2)                          # (B,m2,3)
        knn2 = knn(centers2, xyz, self.k)
        out2 = self.fuse2(centers2, feats, xyz, knn2)               # (B,m2,F)

        # 将 out2 上采样回 N（nearest）
        with torch.no_grad():
            nn2idx_full = self._nearest_index(centers2, xyz)        # (B,N)
        up2 = index_points(out2, nn2idx_full)                       # (B,N,F)

        # 融合 + 主头
        fused = self.back_proj(torch.cat([feats, up2], dim=-1))     # (B,N,F)
        logits = self.main_head(fused)                              # (B,N,C)

        # DEBUG（仅在训练初期）
        if self.training:
            for t, name in [(out1, "out1"), (out2, "out2"), (sem1_logits, "sem1_logits")]:
                nans = torch.isnan(t).any().item()
                infs = torch.isinf(t).any().item()
                if nans or infs:
                    raise RuntimeError(f"[NaN/Inf] {name} has NaN={nans}, Inf={infs}")
        return logits, {"sem1_logits": sem1_logits, "idx1": idx1, "idx2": idx2}

    def get_loss(self,
                 points,          # (B,N,3)
                 labels,          # (B,N)
                 class_weights=None,   # (C,)
                 ignore_index: int = -1,
                 aux_weight: float = 0.4,
                 label_smoothing: float = 0.05,
                 focal_gamma: float = 1.5,  # 设 0 关闭 focal
                 ):
        logits, aux = self.forward(points)  # logits:(B,N,C)
        B, N, C = logits.shape

        # 主损失
        loss_main = F.cross_entropy(
            logits.reshape(-1, C),
            labels.reshape(-1),
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='none'
        )  # (B*N,)
        # focal（可选）
        if focal_gamma and focal_gamma > 0:
            with torch.no_grad():
                pt = torch.softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).clamp_min(1e-6).squeeze(-1)
                mask = (labels != ignore_index)
            loss_main = loss_main * ((1 - pt.reshape(-1)).pow(focal_gamma))
            loss_main = loss_main[mask.reshape(-1)]

        loss_main = loss_main.mean()

        # 辅助损失（在 m1 上对齐标签）
        sem1_logits = aux["sem1_logits"]            # (B,m1,C)
        idx1 = aux["idx1"]                          # (B,m1)
        labels_m1 = torch.gather(labels, 1, idx1)   # (B,m1)
        loss_aux = F.cross_entropy(
            sem1_logits.reshape(-1, C),
            labels_m1.reshape(-1),
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        loss = loss_main + aux_weight * loss_aux

        # 计算训练时的粗 accuracy（不含 ignore）
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            valid = (labels != ignore_index)
            acc = (pred[valid] == labels[valid]).float().mean().item() if valid.any() else 0.0

        return loss, logits, {"train_acc": acc, "loss_main": loss_main.item(), "loss_aux": loss_aux.item()}
