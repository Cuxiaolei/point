import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import DynamicBallQuery, AdaptiveAggregation, CSIT


class DATSeg(nn.Module):
    """动态聚合Transformer点云分割模型（通道匹配版）"""

    def __init__(self, num_classes=3):  # 与配置一致
        super().__init__()
        self.num_classes = num_classes

        # 输入特征处理（3→64通道）
        self.input_conv = nn.Conv1d(3, 64, kernel_size=1, bias=False)

        # 动态球查询与聚合层1（64→128→64）
        self.dynamic_query1 = DynamicBallQuery(min_radius=0.05, max_radius=0.3, num_neighbors=32)
        self.agg1 = AdaptiveAggregation(in_channels=64, out_channels=128)
        self.down1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.reduce1 = nn.Conv1d(128, 64, kernel_size=1, bias=False)  # 降为64通道

        # 动态球查询与聚合层2（64→256→64）
        self.dynamic_query2 = DynamicBallQuery(min_radius=0.1, max_radius=0.5, num_neighbors=32)
        self.agg2 = AdaptiveAggregation(in_channels=64, out_channels=256)
        self.down2 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv1d(256, 64, kernel_size=1, bias=False)  # 降为64通道

        # 跨尺度交互层（输入均为64通道）
        self.csit1 = CSIT(dim=64)
        self.csit2 = CSIT(dim=64)

        # 输出层（64*3=192通道输入）
        self.output_conv = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, points):
        B, N, _ = points.shape
        coords = points[:, :, :3]  # 仅用坐标

        # 输入特征提取
        x0 = self.input_conv(coords.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 64)

        # 下采样1
        center_idx1 = torch.randint(0, N, (B, N // 2), device=coords.device)
        x1_local = self.dynamic_query1(coords, x0, center_idx1)  # (B, N//2, 32, 64)
        x1 = self.agg1(x1_local)  # (B, N//2, 128)
        x1 = self.down1(x1.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N//2, 128)
        x1 = self.reduce1(x1.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N//2, 64)

        # 下采样2
        center_idx2 = torch.randint(0, N//2, (B, N//4), device=coords.device)
        x2_local = self.dynamic_query2(coords, x1, center_idx2)  # (B, N//4, 32, 64)
        x2 = self.agg2(x2_local)  # (B, N//4, 256)
        x2 = self.down2(x2.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N//4, 256)
        x2 = self.reduce2(x2.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N//4, 64)

        # 跨尺度交互
        x1_up = F.interpolate(x1.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, 64)
        x1_enhanced = self.csit1(x0, x1_up)  # 均为64通道

        x2_up = F.interpolate(x2.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, 64)
        x0_enhanced = self.csit2(x1_enhanced, x2_up)  # 均为64通道

        # 特征融合
        x2_up_full = F.interpolate(x2.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, 64)
        fused = torch.cat([x0_enhanced, x1_up, x2_up_full], dim=-1)  # (B, N, 192)
        logits = self.output_conv(fused.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, num_classes)

        return logits

    def get_loss(self, points, labels, ignore_index=-1):
        logits = self.forward(points)
        loss = F.cross_entropy(
            logits.transpose(1, 2),  # (B, C, N)
            labels,
            ignore_index=ignore_index
        )

        # 计算准确率
        mask = labels != ignore_index
        preds = torch.argmax(logits, dim=-1)
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

        return loss, logits  # 修正：返回loss和logits（原代码返回值错误）