# sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules_sgdat import DynamicRadiusChannelFusion, ChannelCCC, LinearSpatialGVA

class SGDATSeg(nn.Module):
    def __init__(self, num_classes=13, input_dim=9, base_dim=64, max_points=8000):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.max_points = max_points

        # 编码器：处理输入的9维特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )

        # 后续的融合层
        self.fuse1 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32)
        self.fuse2 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32)

        self.aux_sem = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, num_classes),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(),
            nn.Linear(base_dim//2, num_classes)
        )

    def forward(self, x):
        B, N, _ = x.shape
        feats = self.encoder(x)  # (B, N, base_dim)

        # 通过融合层处理特征
        out1, knn_idx1 = self.fuse1(x, feats, center_idx=x)
        out2, knn_idx2 = self.fuse2(x, feats, center_idx=x)

        # 合并处理后的特征
        fused_feats = out1 + out2
        logits = self.classifier(fused_feats)  # (B, N, num_classes)
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
