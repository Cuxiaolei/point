# models/sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules_sgdat import DynamicRadiusChannelFusion, ChannelCCC, LinearSpatialGVA, farthest_point_sample

class SGDATSeg(nn.Module):
    """
    SGDATSeg 模型（重构版）
    - 支持 aux supervision（语义引导分支）
    - get_loss 支持 class_weights 与 aux loss 权重
    - debug 模式下会检查中间张量的 NaN/Inf
    """
    def __init__(self, num_classes=13, input_dim=9, base_dim=64, max_points=8000,
                 sample_m1=None, sample_m2=None, aux_weight=0.3, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.max_points = max_points
        self.aux_weight = aux_weight
        self.debug = debug

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        # dynamic fusion stages
        self.fuse1 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=16, min_radius=0.02, max_radius=0.15)
        self.fuse2 = DynamicRadiusChannelFusion(in_channels=base_dim, out_channels=base_dim, num_neighbors=16, min_radius=0.05, max_radius=0.3)

        # aux semantic branch -> output logits (不要 Softmax)
        self.aux_sem = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim//2, self.num_classes)
        )

        # channel and spatial modules
        self.channel_ccc = ChannelCCC(dim=base_dim)
        self.linear_gva = LinearSpatialGVA(dim=base_dim)

        # upsample & fuse
        self.up_mlp = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim//2, self.num_classes)
        )

        # sampling sizes override
        self.sample_m1 = sample_m1
        self.sample_m2 = sample_m2

    def forward(self, x, return_aux=False):
        """
        x: (B,N,input_dim)
        return_aux: if True, return (logits, aux_dict)
        aux_dict contains:
            - idx1, idx2: center indices (B, m1)/(B,m2)
            - sem1_logits, sem2_logits: logits on centers
        """
        B, N, D = x.shape
        coords = x[..., :3]

        feats = self.encoder(x)  # (B,N,base_dim)
        if self.debug:
            n_nan = torch.isnan(feats).sum().item()
            n_inf = torch.isinf(feats).sum().item()
            if n_nan or n_inf:
                print(f"[DEBUG] encoder NaN={n_nan} Inf={n_inf}")
                raise RuntimeError("NaN/Inf in encoder output")

        # determine sample sizes
        m1 = self.sample_m1 if self.sample_m1 is not None else min(512, max(32, N // 8))
        m2 = self.sample_m2 if self.sample_m2 is not None else min(128, max(16, N // 32))
        m1 = min(m1, N)
        m2 = min(m2, N)

        # fps (with fallback to random)
        with torch.no_grad():
            try:
                idx1 = farthest_point_sample(coords, m1)  # (B,m1)
                idx2 = farthest_point_sample(coords, m2)  # (B,m2)
            except Exception:
                idx1 = torch.randint(0, N, (B, m1), device=x.device)
                idx2 = torch.randint(0, N, (B, m2), device=x.device)

        # fuse stages
        out1, knn1 = self.fuse1(coords, feats, center_idx=idx1)  # (B,m1,base_dim)
        sem1_logits = self.aux_sem(out1)  # (B,m1,num_classes)

        out2, knn2 = self.fuse2(coords, feats, center_idx=idx2)  # (B,m2,base_dim)
        sem2_logits = self.aux_sem(out2)  # (B,m2,num_classes)

        if self.debug:
            for name, t in [("out1", out1), ("out2", out2), ("sem1_logits", sem1_logits)]:
                print(f"[DEBUG] {name} shape: {t.shape}, NaN={torch.isnan(t).sum().item()}, Inf={torch.isinf(t).sum().item()}")

        out1_ccc = self.channel_ccc(out1)
        out2_ccc = self.channel_ccc(out2)

        out1_gva = self.linear_gva(out1_ccc)
        out2_gva = self.linear_gva(out2_ccc)

        # upsample to N using nearest interpolation
        out1_up = F.interpolate(out1_gva.permute(0,2,1), size=N, mode='nearest').permute(0,2,1)
        out2_up = F.interpolate(out2_gva.permute(0,2,1), size=N, mode='nearest').permute(0,2,1)

        fused = torch.cat([out1_up, out2_up], dim=-1)  # (B,N,2*D)
        fused = self.up_mlp(fused)  # (B,N,base_dim)

        logits = self.classifier(fused)  # (B,N,num_classes)

        if self.debug:
            n_nan = torch.isnan(logits).sum().item()
            n_inf = torch.isinf(logits).sum().item()
            if n_nan or n_inf:
                print(f"[DEBUG] logits NaN={n_nan} Inf={n_inf}")
                raise RuntimeError("NaN/Inf in logits")

        if return_aux:
            aux = {
                'idx1': idx1,
                'idx2': idx2,
                'sem1_logits': sem1_logits,
                'sem2_logits': sem2_logits
            }
            return logits, aux
        else:
            return logits

    def get_loss(self, points, labels, ignore_index=-1, class_weights=None, aux_weight=None):
        """
        Compute total loss:
            loss_main = CE over full logits
            loss_aux1 = CE over sem1 logits (gather GT by idx1)
            loss_aux2 = CE over sem2 logits (gather GT by idx2)
        aux_weight: if None use self.aux_weight
        Returns: loss, logits
        """
        if aux_weight is None:
            aux_weight = self.aux_weight

        # forward with aux outputs
        logits, aux = self.forward(points, return_aux=True)

        # numeric checks
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            n_nan = torch.isnan(logits).sum().item()
            n_inf = torch.isinf(logits).sum().item()
            print(f"[NUMERIC] logits NaN={n_nan}, Inf={n_inf}")
            raise RuntimeError("Logits contain NaN/Inf")

        # main loss (use class_weights if provided)
        if class_weights is not None:
            loss_main = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=ignore_index, weight=class_weights)
        else:
            loss_main = F.cross_entropy(logits.transpose(1,2), labels, ignore_index=ignore_index)

        # auxiliary losses: gather labels at sampled centers
        loss_aux_total = 0.0
        try:
            idx1 = aux['idx1']  # (B,m1)
            idx2 = aux['idx2']  # (B,m2)
            sem1_logits = aux['sem1_logits']  # (B,m1,num_classes)
            sem2_logits = aux['sem2_logits']  # (B,m2,num_classes)

            # gather labels for idx1/idx2
            # labels: (B,N)
            labels_centers1 = labels.gather(1, idx1)  # (B,m1)
            labels_centers2 = labels.gather(1, idx2)  # (B,m2)

            if class_weights is not None:
                loss_aux1 = F.cross_entropy(sem1_logits.transpose(1,2), labels_centers1, ignore_index=ignore_index, weight=class_weights)
                loss_aux2 = F.cross_entropy(sem2_logits.transpose(1,2), labels_centers2, ignore_index=ignore_index, weight=class_weights)
            else:
                loss_aux1 = F.cross_entropy(sem1_logits.transpose(1,2), labels_centers1, ignore_index=ignore_index)
                loss_aux2 = F.cross_entropy(sem2_logits.transpose(1,2), labels_centers2, ignore_index=ignore_index)

            loss_aux_total = loss_aux1 + loss_aux2
        except Exception as e:
            # 如果 aux 计算失败（例如 idx gather 异常），打印错误但不阻断训练（改为只用主 loss）
            print("[WARNING] aux loss computation failed:", str(e))
            loss_aux_total = 0.0

        total_loss = loss_main + aux_weight * loss_aux_total
        return total_loss, logits
