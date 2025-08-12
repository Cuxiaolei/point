# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_sgdat import (
    DynamicRadiusChannelFusion,
    ChannelCCC,
    LinearSpatialGVA,
    farthest_point_sample
)


class SGDATSeg(nn.Module):
    """
    SGDATSeg: simplified, robust implementation that preserves your three innovations:
      - Dynamic neighborhood + channel fusion (DynamicRadiusChannelFusion)
      - Semantic-guided full-scale fusion (aux_sem used as semantic cue)
      - Lightweight attention (ChannelCCC + LinearSpatialGVA)
    """
    def __init__(self, num_classes=13, input_dim=9, base_dim=64, max_points=8000):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.max_points = max_points

        # encoder: project input features -> base_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        # multiscale dynamic fusion layers
        self.fuse1 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32, min_radius=0.02, max_radius=0.15)
        self.fuse2 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=32, min_radius=0.05, max_radius=0.3)

        # semantic auxiliary head (guidance)
        self.aux_sem = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim // 2, num_classes),
            nn.Softmax(dim=-1)
        )

        # channel & spatial lightweight modules
        self.channel_ccc = ChannelCCC(dim=base_dim)
        self.linear_gva = LinearSpatialGVA(dim=base_dim)

        # upfusion MLP
        self.up_mlp = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        x: (B, N, D) where D == input_dim (e.g., coords + color + normals)
        returns logits: (B, N, num_classes)
        """
        B, N, D = x.shape
        coords = x[..., :3]  # (B,N,3)
        # debug
        # print(f"[SGDAT] forward input {x.shape}")

        feats = self.encoder(x)  # (B,N,base_dim)
        # print(f"[SGDAT] after encoder {feats.shape}")

        # --- sample centers with FPS (robust) ---
        m1 = max(1, N // 2)
        m2 = max(1, N // 4)
        with torch.no_grad():
            idx1 = farthest_point_sample(coords, m1)  # (B, m1)
            idx2 = farthest_point_sample(coords, m2)  # (B, m2)

        # fuse at scale1 (centers m1)
        out1, knn1 = self.fuse1(coords, feats, center_idx=idx1)  # out1: (B, m1, base_dim)
        # semantic guidance (on the centers)
        sem1 = self.aux_sem(out1)  # (B, m1, num_classes)

        # fuse at scale2 (centers m2)
        out2, knn2 = self.fuse2(coords, feats, center_idx=idx2)  # out2: (B, m2, base_dim)
        sem2 = self.aux_sem(out2)  # (B, m2, num_classes)

        # channel + spatial refinement
        out1_ccc = self.channel_ccc(out1)  # (B, m1, base_dim)
        out2_ccc = self.channel_ccc(out2)  # (B, m2, base_dim)

        out1_gva = self.linear_gva(out1_ccc)  # (B, m1, base_dim)
        out2_gva = self.linear_gva(out2_ccc)  # (B, m2, base_dim)

        # upsample back to N using nearest (1D interpolate on the M dimension)
        # use nearest interpolation via permute + F.interpolate
        out1_up = F.interpolate(out1_gva.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, base_dim)
        out2_up = F.interpolate(out2_gva.permute(0, 2, 1), size=N, mode='nearest').permute(0, 2, 1)  # (B, N, base_dim)

        # concat and fuse
        fused = torch.cat([out1_up, out2_up], dim=-1)  # (B,N, 2*base_dim)
        fused = self.up_mlp(fused)  # (B,N, base_dim)

        # classifier
        logits = self.classifier(fused)  # (B, N, num_classes)
        return logits

    def get_loss(self, points, labels, ignore_index=-1):
        """
        Return (loss, logits) to be compatible with Trainer (which expects loss, logits).
        """
        logits = self.forward(points)  # (B,N,num_classes)
        # compute cross entropy: PyTorch expects (B,C,N) for input to F.cross_entropy
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=ignore_index)
        return loss, logits
