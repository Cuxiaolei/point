# sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules_sgdat import DynamicRadiusChannelFusion, ChannelCCC, LinearSpatialGVA

class SGDATSeg(nn.Module):
    """
    Semantic-Guided Dynamic Aggregation Transformer for point cloud segmentation.
    输入: points (B,N,3) 或 (B,N,3+feat)
    输出: logits (B,N,num_classes)
    """
    def __init__(self, num_classes=13, input_dim=3, base_dim=64, max_points=8000):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.base_dim = base_dim

        # 初始编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )

        # 两级动态融合下采样
        self.fuse1 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32, min_radius=0.02, max_radius=0.15)
        self.fuse2 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32, min_radius=0.05, max_radius=0.3)

        # 语义提示支路
        self.aux_sem = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, num_classes),
            nn.Softmax(dim=-1)
        )

        # 使用 ChannelCCC 和 LinearSpatialGVA
        self.channel_ccc = ChannelCCC(dim=base_dim)  # 用于特征通道加权
        self.linear_gva = LinearSpatialGVA(dim=base_dim)  # 用于空间聚合

        # 上采样融合
        self.up_mlp = nn.Sequential(
            nn.Linear(base_dim * 3, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, num_classes)
        )

    def forward(self, x):
        B, N, _ = x.shape
        coords = x[..., :3]

        feats = self.encoder(x)  # (B,N,base_dim)

        # sample centers indices
        idx1 = torch.randint(0, N, (B, N//2), device=x.device)
        x1, knn1 = self.fuse1(coords, feats, idx1)  # (B, N//2, D)
        sem1 = self.aux_sem(x1)  # (B, N//2, num_classes) soft cues

        idx2 = torch.randint(0, N//2, (B, N//4), device=x.device)
        x2, knn2 = self.fuse2(coords, x1, idx2)  # (B, N//4, D)
        sem2 = self.aux_sem(x2)  # (B, N//4, num_classes)

        # 使用 ChannelCCC 对特征进行加权
        x1_ccc = self.channel_ccc(x1)  # (B, N//2, base_dim)
        x2_ccc = self.channel_ccc(x2)  # (B, N//4, base_dim)

        # 使用 LinearSpatialGVA 进行空间特征聚合
        x1_gva = self.linear_gva(x1_ccc)  # (B, N//2, base_dim)
        x2_gva = self.linear_gva(x2_ccc)  # (B, N//4, base_dim)

        # 上采样并融合（特征拼接）
        x1_up = F.interpolate(x1_gva.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, base_dim)
        x2_up = F.interpolate(x2_gva.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)

        # 特征融合
        fused = torch.cat([x1_up, x2_up], dim=-1)  # (B, N, base_dim * 2)
        fused = self.up_mlp(fused)  # (B, N, base_dim)

        # 最终分类输出
        logits = self.classifier(fused)  # (B, N, num_classes)
        return logits

    def get_loss(self, points, labels, ignore_index=-1):
        logits = self.forward(points)
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=ignore_index)
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_index
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        return loss, acc
