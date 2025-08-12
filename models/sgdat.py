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
    def __init__(self, num_classes=13, input_dim=9, base_dim=64, max_points=8000,
                 sample_m1=None, sample_m2=None, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.max_points = max_points
        self.debug = debug

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        self.fuse1 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim,
                                                num_neighbors=16, min_radius=0.02, max_radius=0.15)
        self.fuse2 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim,
                                                num_neighbors=16, min_radius=0.05, max_radius=0.3)

        self.aux_sem = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim//2, num_classes),
            nn.Softmax(dim=-1)
        )

        self.channel_ccc = ChannelCCC(dim=base_dim)
        self.linear_gva = LinearSpatialGVA(dim=base_dim)

        self.up_mlp = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim//2, num_classes)
        )

        # allow override sample sizes (for experiments)
        self.sample_m1 = sample_m1
        self.sample_m2 = sample_m2

    def forward(self, x):
        B, N, D = x.shape
        coords = x[..., :3]

        # encode
        feats = self.encoder(x)  # (B,N,base_dim)
        if self.debug:
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print("[DEBUG] encoder produced NaN/Inf", torch.isnan(feats).sum().item(), torch.isinf(feats).sum().item())
                raise RuntimeError("NaN in encoder output")

        # choose sample counts but keep them modest to save compute:
        if self.sample_m1 is not None:
            m1 = min(self.sample_m1, N)
        else:
            m1 = min(512, max(32, N // 8))  # default conservative
        if self.sample_m2 is not None:
            m2 = min(self.sample_m2, N)
        else:
            m2 = min(128, max(16, N // 32))

        # FPS (if npoint small enough)
        with torch.no_grad():
            try:
                idx1 = farthest_point_sample(coords, m1)  # (B,m1)
                idx2 = farthest_point_sample(coords, m2)  # (B,m2)
            except Exception as e:
                # fallback: random sampling (much faster)
                idx1 = torch.randint(0, N, (B, m1), device=x.device)
                idx2 = torch.randint(0, N, (B, m2), device=x.device)

        # fuse1
        out1, knn1 = self.fuse1(coords, feats, center_idx=idx1)  # (B,m1,base_dim)
        sem1 = self.aux_sem(out1)  # (B,m1,num_classes)

        out2, knn2 = self.fuse2(coords, feats, center_idx=idx2)  # (B,m2,base_dim)
        sem2 = self.aux_sem(out2)

        if self.debug:
            if torch.isnan(out1).any() or torch.isinf(out1).any():
                print("[DEBUG] out1 NaN/Inf", torch.isnan(out1).sum().item(), torch.isinf(out1).sum().item())
                raise RuntimeError("NaN in out1")
            if torch.isnan(out2).any() or torch.isinf(out2).any():
                print("[DEBUG] out2 NaN/Inf", torch.isnan(out2).sum().item(), torch.isinf(out2).sum().item())
                raise RuntimeError("NaN in out2")

        out1_ccc = self.channel_ccc(out1)
        out2_ccc = self.channel_ccc(out2)

        out1_gva = self.linear_gva(out1_ccc)
        out2_gva = self.linear_gva(out2_ccc)

        # upsample back to N (nearest)
        out1_up = F.interpolate(out1_gva.permute(0,2,1), size=N, mode='nearest').permute(0,2,1)
        out2_up = F.interpolate(out2_gva.permute(0,2,1), size=N, mode='nearest').permute(0,2,1)

        fused = torch.cat([out1_up, out2_up], dim=-1)
        fused = self.up_mlp(fused)

        if self.debug:
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                print("[DEBUG] fused NaN/Inf", torch.isnan(fused).sum().item(), torch.isinf(fused).sum().item())
                raise RuntimeError("NaN in fused")

        logits = self.classifier(fused)  # (B,N,num_classes)
        return logits

    def get_loss(self, points, labels, ignore_index=-1):
        logits = self.forward(points)
        # numeric check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # print some diagnostics (first batch only)
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            print(f"[NUMERIC] logits has NaN={nan_count}, Inf={inf_count}")
            # return a high loss to stop training, or raise
            raise RuntimeError("Logits contain NaN/Inf")
        loss = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=ignore_index)
        return loss, logits
