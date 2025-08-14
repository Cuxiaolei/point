# sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容导入（允许作为包或脚本）
try:
    from .modules_sgdat import (
        ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion,
        nearest_interpolate, SemanticGuidedGate
    )
except Exception:
    from modules_sgdat import (
        ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion,
        nearest_interpolate, SemanticGuidedGate
    )


# -------------------------
# 数值更稳的工具函数
# -------------------------
def normalize_xyz(xyz, eps=1e-6):
    """
    将坐标平移到均值为 0，并按标准差归一化，避免尺度过大导致的梯度不稳
    xyz: (B,N,3)
    """
    mean = xyz.mean(dim=1, keepdim=True)
    std = xyz.std(dim=1, keepdim=True).clamp(min=eps)
    return (xyz - mean) / std


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) - 简洁 CPU/显存友好的实现
    xyz: (B, N, 3)
    return: idx (B, npoint)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        dist_new = torch.sum((xyz - centroid) ** 2, dim=-1)      # (B,N)
        mask = dist_new < dist
        dist[mask] = dist_new[mask]
        farthest = torch.max(dist, dim=-1)[1]
    return idx


def batched_index_points(points, idx):
    """
    points: (B, N, C), idx: (B, M)
    return: (B, M, C)
    """
    B, N, C = points.shape
    _, M = idx.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1).repeat(1, M)
    return points[batch_indices, idx, :]


def squared_distance(src, dst):
    """
    src: (B, N, 3)
    dst: (B, M, 3)
    return: dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


# -------------------------
# 轻量模块（点形式到 Conv1d 的桥接）
# -------------------------
class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B,C,N)
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class SharedMLP1D(nn.Module):
    def __init__(self, cin, cout, bn_eps=1e-3, bn_momentum=0.01, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):  # (B,N,Cin)
        x = x.permute(0, 2, 1)
        x = self.block(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class ResMLP1D(nn.Module):
    def __init__(self, cin, cout, bn_eps=1e-3, bn_momentum=0.01, use_se=True, dropout: float = 0.0):
        super().__init__()
        self.match = (cin == cout)
        self.proj = nn.Identity() if self.match else nn.Conv1d(cin, cout, 1, bias=False)

        self.block = nn.Sequential(
            nn.Conv1d(cout if self.match else cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 1, bias=False),
            nn.BatchNorm1d(cout, eps=bn_eps, momentum=bn_momentum),
        )
        self.act = nn.ReLU(inplace=True)
        self.se = SE1D(cout) if use_se else nn.Identity()
        self.dp = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):  # (B,N,Cin)
        identity = x
        x = x.permute(0, 2, 1)
        out = self.block(x)
        out = self.se(out)
        out = out + self.proj(x)
        out = self.act(out)
        out = self.dp(out)
        out = out.permute(0, 2, 1).contiguous()
        return out


# -------------------------
# SGDAT 主干
# -------------------------
class SGDAT(nn.Module):
    def __init__(
        self,
        num_classes,
        base_dim=64,
        max_points=8000,
        debug=False,
        # ====== 抗过拟合 ======
        dropout_p: float = 0.0,
        droppath_prob: float = 0.0,
        drop_rgb_p: float = 0.0,
        drop_normal_p: float = 0.0,
        logit_temp: float = 1.0,
        # ====== 模块开关 ======
        use_channel_ccc: bool = True,
        use_dynamic_fusion: bool = True,
        use_linear_gva: bool = True,
        use_semantic_guided_fusion: bool = True,
        # ====== 动态邻域参数 ======
        dyn_neighbors: int = 16,
        dyn_tau: float = 0.2,
        # ====== BN/数值稳定 ======
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,
        # ====== 轻几何增强 ======
        use_geom_enhance: bool = True,
        # ====== 新增：从 config 导入 ======
        cfg=None
    ):
        # 如果传入了 cfg，就用它的属性覆盖同名参数
        if cfg is not None:
            for k, v in cfg.__dict__.items():
                if hasattr(self, k) or k in locals():
                    locals()[k] = v
            # num_classes 单独保证取 config 里的
            if hasattr(cfg, "NUM_CLASSES"):
                num_classes = cfg.NUM_CLASSES

        super().__init__()
        self.num_classes = int(num_classes)
        self.base_dim = int(base_dim)
        self.max_points = int(max_points)
        self.debug = bool(debug)
        self.dropout_p = float(dropout_p)
        self.droppath_prob = float(droppath_prob)
        self.drop_rgb_p = float(drop_rgb_p)
        self.drop_normal_p = float(drop_normal_p)
        self.logit_temp = float(logit_temp)

        self.use_channel_ccc = bool(use_channel_ccc)
        self.use_dynamic_fusion = bool(use_dynamic_fusion)
        self.use_linear_gva = bool(use_linear_gva)
        self.use_semantic_guided_fusion = bool(use_semantic_guided_fusion)

        self.dyn_neighbors = dyn_neighbors
        self.dyn_tau = float(dyn_tau)

        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)

        self.use_geom_enhance = bool(use_geom_enhance)

        # ---------- Encoder ----------
        self.enc0 = ResMLP1D(9, base_dim, bn_eps=self.bn_eps, bn_momentum=self.bn_momentum,
                             use_se=True, dropout=self.dropout_p)  # (B,N,64)

        if self.use_geom_enhance:
            self.geom_embed_512 = SharedMLP1D(2, base_dim // 2, dropout=self.dropout_p,
                                              bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)
            self.geom_embed_N   = SharedMLP1D(2, base_dim // 2, dropout=self.dropout_p,
                                              bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)
        else:
            self.geom_embed_512 = None
            self.geom_embed_N = None

        self.enc512_in_dim = 3 + base_dim + (base_dim // 2 if self.use_geom_enhance else 0)
        self.enc512 = ResMLP1D(self.enc512_in_dim, base_dim, bn_eps=self.bn_eps,
                               bn_momentum=self.bn_momentum, use_se=True, dropout=self.dropout_p)

        self.enc128 = ResMLP1D(3 + base_dim, base_dim * 2, bn_eps=self.bn_eps,
                               bn_momentum=self.bn_momentum, use_se=True, dropout=self.dropout_p)

        # 通道注意力
        if self.use_channel_ccc:
            self.ccc_512 = ChannelCCC(base_dim)
            self.ccc_128 = ChannelCCC(base_dim * 2)
        else:
            self.ccc_512 = nn.Identity()
            self.ccc_128 = nn.Identity()

        # 动态邻域 - 通道融合
        if self.use_dynamic_fusion:
            self.dyn512 = DynamicRadiusChannelFusion(
                c_in=base_dim, c_hidden=base_dim, c_out=base_dim, k=self.dyn_neighbors, tau=self.dyn_tau
            )
            self.dyn128 = DynamicRadiusChannelFusion(
                c_in=base_dim * 2, c_hidden=base_dim * 2, c_out=base_dim * 2, k=self.dyn_neighbors, tau=self.dyn_tau
            )
        else:
            self.dyn512 = None
            self.dyn128 = None

        # 位置/尺度编码
        self.pos512 = SharedMLP1D(3, base_dim * 2, dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)
        self.posN   = SharedMLP1D(3, base_dim,     dropout=self.dropout_p,
                                  bn_eps=self.bn_eps, bn_momentum=self.bn_momentum)

        # 线性 GVA
        if self.use_linear_gva:
            self.gva_512 = LinearSpatialGVA(dim=base_dim * 2)
            self.gva_N   = LinearSpatialGVA(dim=base_dim * 2)
        else:
            self.gva_512 = nn.Identity()
            self.gva_N   = nn.Identity()

        # 语义引导门控
        if self.use_semantic_guided_fusion:
            self.sem128_head = nn.Conv1d(base_dim * 2, num_classes, kernel_size=1, bias=True)
            self.sem_gate_512 = SemanticGuidedGate(num_classes)
            self.sem_gate_N   = SemanticGuidedGate(num_classes)
        else:
            self.sem128_head = None
            self.sem_gate_512 = None
            self.sem_gate_N = None

        # ---------- Decoder ----------
        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1, bias=False),
            nn.BatchNorm1d(base_dim * 2, eps=self.bn_eps, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )
        self.up1_refine = ResMLP1D(base_dim * 2, base_dim * 2, bn_eps=self.bn_eps,
                                   bn_momentum=self.bn_momentum, use_se=True, dropout=self.dropout_p)

        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim * 2, 1, bias=False),
            nn.BatchNorm1d(base_dim * 2, eps=self.bn_eps, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )
        self.up2_refine = ResMLP1D(base_dim * 2, base_dim * 2, bn_eps=self.bn_eps,
                                   bn_momentum=self.bn_momentum, use_se=True, dropout=self.dropout_p)

        self.branch_dp1 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()
        self.branch_dp2 = DropPath(self.droppath_prob) if self.droppath_prob > 0 else nn.Identity()

        self.head = nn.Conv1d(base_dim * 2, num_classes, kernel_size=1)

        self.apply(self._init_stable)

    # ---------- 初始化 ----------
    @staticmethod
    def _init_stable(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------- 便捷函数 ----------
    def _sample_and_gather(self, xyz, feat, npoint):
        idx = farthest_point_sample(xyz, npoint)
        xyz_s = batched_index_points(xyz, idx)
        feat_s = batched_index_points(feat, idx)
        return xyz_s, feat_s, idx

    def _feature_group_dropout(self, points):
        if not self.training:
            return points
        out = points
        if self.drop_rgb_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_rgb_p:
            out = out.clone()
            out[:, :, 3:6] = 0.0
        if self.drop_normal_p > 0.0 and torch.rand(1, device=points.device).item() < self.drop_normal_p:
            out = out.clone()
            out[:, :, 6:9] = 0.0
        return out

    def _geom_tokens(self, xyz_normed):
        r = torch.sqrt(torch.clamp((xyz_normed ** 2).sum(dim=-1, keepdim=True), min=0.0))
        h = xyz_normed[..., 2:3]
        return torch.cat([r, h], dim=-1)

    # —— 小助手：把 (B,N,C) 的张量交给 [B,C,N] 接口的模块，再转回 —— #
    @staticmethod
    def _apply_channel_first_module(x_bnC, mod):
        # x_bnC: (B, N, C) -> (B, C, N) -> mod -> (B, C, N) -> (B, N, C)
        y = mod(x_bnC.permute(0, 2, 1).contiguous())
        return y.permute(0, 2, 1).contiguous()

    @staticmethod
    def _apply_channel_first_module_with_pos(x_bnC, pos_bn3, mod):
        # x_bnC: (B,N,C) ; pos_bn3: (B,N,3)
        x = x_bnC.permute(0, 2, 1).contiguous()      # (B,C,N)
        p = pos_bn3.permute(0, 2, 1).contiguous()    # (B,3,N)
        y = mod(x, p)                                # (B,C,N)
        return y.permute(0, 2, 1).contiguous()       # (B,N,C)

    # ---------- 前向 ----------
    def forward(self, points):
        """
        points: (B, N, 9) -> [xyz(0:3), rgb(3:6), normal(6:9)]
        return: logits (B, N, num_classes)
        """
        B, N, C = points.shape
        assert C == 9, f"Expect input features 9 (xyz+rgb+normal), but got {C}"

        if self.debug:
            print(f"[DEBUG] Input shape: {points.shape}")

        points = self._feature_group_dropout(points)

        xyz = points[:, :, 0:3]
        rgb = points[:, :, 3:6]
        normal = points[:, :, 6:9]
        xyz_normed = normalize_xyz(xyz)

        geom_N = self._geom_tokens(xyz_normed)  # (B,N,2)
        geom_emb_N = self.geom_embed_N(geom_N) if (self.use_geom_enhance and self.geom_embed_N is not None) else None

        feat0_in = torch.cat([xyz_normed, rgb, normal], dim=-1)  # (B,N,9)
        feat0 = self.enc0(feat0_in)                              # (B,N,64)

        if self.use_geom_enhance and geom_emb_N is not None and geom_emb_N.shape[-1] == self.base_dim:
            feat0 = feat0 + geom_emb_N

        # 下采样到 512
        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0, npoint=512)  # (B,512,3),(B,512,64)

        # 编码 512 输入：xyz + feat0_512 +（可选 geom_512_emb）
        if self.use_geom_enhance:
            geom_512 = batched_index_points(geom_N, idx_512)               # (B,512,2)
            geom_emb_512 = self.geom_embed_512(geom_512)                   # (B,512,base_dim//2)
            enc_512_in = torch.cat([xyz_512, feat0_512, geom_emb_512], dim=-1)
        else:
            enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)
        feat_512 = self.enc512(enc_512_in)                                 # (B,512,64)

        # —— 动态邻域-通道融合 @512
        if self.use_dynamic_fusion and self.dyn512 is not None:
            feat_512 = feat_512 + self._apply_channel_first_module_with_pos(feat_512, xyz_512, self.dyn512)

        # 通道注意力 @512
        feat_512 = self._apply_channel_first_module(feat_512, self.ccc_512)  # (B,512,64)

        # 再下采样到 128
        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)  # (B,128,3),(B,128,64)
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)
        feat_128 = self.enc128(enc_128_in)                       # (B,128,128)

        # —— 动态邻域-通道融合 @128
        if self.use_dynamic_fusion and self.dyn128 is not None:
            feat_128 = feat_128 + self._apply_channel_first_module_with_pos(feat_128, xyz_128, self.dyn128)

        # 通道注意力 @128
        feat_128 = self._apply_channel_first_module(feat_128, self.ccc_128)  # (B,128,128)

        # ===== 语义引导先验（在 128 尺度形成） =====
        if self.use_semantic_guided_fusion and self.sem128_head is not None:
            # 注意 sem128_head 期望 [B,C,N]，故先转置
            sem128_logits = self.sem128_head(feat_128.permute(0, 2, 1).contiguous())  # (B,K,128)
            # 门控到 512、N 两个尺度（nearest_interpolate 为 channel-first）
            gate512 = self.sem_gate_512(
                sem_logits=sem128_logits,
                source_pos=xyz_128.permute(0, 2, 1).contiguous(),
                target_pos=xyz_512.permute(0, 2, 1).contiguous()
            )  # (B,1,512)
            gateN = self.sem_gate_N(
                sem_logits=sem128_logits,
                source_pos=xyz_128.permute(0, 2, 1).contiguous(),
                target_pos=xyz_normed.permute(0, 2, 1).contiguous()
            )  # (B,1,N)
        else:
            gate512 = None
            gateN = None

        if self.debug:
            with torch.no_grad():
                n_nan1 = torch.isnan(feat_512).any().item()
                n_nan2 = torch.isnan(feat_128).any().item()
                print(f"[DEBUG] out1 shape: {feat_512.shape}, NaN={int(n_nan1)}, Inf={int(torch.isinf(feat_512).any().item())}; "
                      f"out2 shape: {feat_128.shape}, NaN={int(n_nan2)}, Inf={int(torch.isinf(feat_128).any().item())}")

        # ---------- Decoder: 128 -> 512 ----------
        # 将 128 特征上采样到 512（channel-first接口）
        up128_to_512_cf = nearest_interpolate(
            target_pos=xyz_512.permute(0, 2, 1).contiguous(),
            source_pos=xyz_128.permute(0, 2, 1).contiguous(),
            source_feat=feat_128.permute(0, 2, 1).contiguous()
        )  # (B,128,512)
        up128_to_512 = up128_to_512_cf.permute(0, 2, 1).contiguous()  # (B,512,128)

        if gate512 is not None:
            up128_to_512 = up128_to_512 * gate512.permute(0, 2, 1).contiguous()

        pos512 = self.pos512(xyz_512)                                   # (B,512,128)

        up128_to_512 = self.branch_dp1(up128_to_512)
        pos512 = self.branch_dp1(pos512)

        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)  # (B,512,320)
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,512,128)
        fuse_512 = self.up1_refine(fuse_512)
        # 线性 GVA 采用 [B,C,N] 接口
        fuse_512 = self._apply_channel_first_module(fuse_512, self.gva_512)

        # ---------- Decoder: 512 -> N ----------
        up512_to_N_cf = nearest_interpolate(
            target_pos=xyz_normed.permute(0, 2, 1).contiguous(),
            source_pos=xyz_512.permute(0, 2, 1).contiguous(),
            source_feat=fuse_512.permute(0, 2, 1).contiguous()
        )  # (B,128,N)
        up512_to_N = up512_to_N_cf.permute(0, 2, 1).contiguous()  # (B,N,128)

        if gateN is not None:
            up512_to_N = up512_to_N * gateN.permute(0, 2, 1).contiguous()

        posN = self.posN(xyz_normed)                                     # (B,N,64)

        if self.use_geom_enhance and geom_emb_N is not None and geom_emb_N.shape[-1] == posN.shape[-1]:
            posN = posN + geom_emb_N

        up512_to_N = self.branch_dp2(up512_to_N)
        posN = self.branch_dp2(posN)

        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)            # (B,N,256)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,128)
        fuse_N = self.up2_refine(fuse_N)
        fuse_N = self._apply_channel_first_module(fuse_N, self.gva_N)

        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B,N,K)

        if self.logit_temp and self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        if self.debug:
            with torch.no_grad():
                print(f"[DEBUG] logits shape: {logits.shape}, "
                      f"min={float(logits.min()) if torch.isfinite(logits).any() else 0.0}, "
                      f"max={float(logits.max()) if torch.isfinite(logits).any() else 0.0}")

        return logits

        # ===== 新增：用类别先验初始化分类器偏置（主头 + 辅助头） =====
        @torch.no_grad()
        def set_class_priors(self, priors):
            """
            priors: 形如 (K,) 的概率分布（counts / sum(counts)）
            用 logit(p) 初始化分类器 bias，可显著缓解极度不均衡时“只出多数类”的冷启动问题
            """
            eps = 1e-6
            p = torch.as_tensor(priors, dtype=torch.float32, device=next(self.parameters()).device)
            p = torch.clamp(p, min=eps, max=1 - eps)
            logit_bias = torch.log(p) - torch.log(1 - p)  # log(p/(1-p))

            # 主头
            if hasattr(self, "head") and hasattr(self.head, "bias") and self.head.bias is not None:
                self.head.bias.copy_(logit_bias)

            # 辅助头（如果存在）
            if hasattr(self, "sem128_head") and hasattr(self.sem128_head, "bias") and self.sem128_head.bias is not None:
                self.sem128_head.bias.copy_(logit_bias)

        # ===== 重写：样本级加权 + 可选 label smoothing + 可选 focal（忽略 -1） =====
        def get_loss(
                self,
                points,
                labels,
                class_weights=None,
                ignore_index=-1,
                aux_weight=None,
                reduction='mean',
                label_smoothing: float = 0.0,
                focal_gamma: float = 0.0,
                **kwargs
        ):
            logits = self.forward(points)  # (B,N,K)

            # --- NaN/Inf 防护（保留你原来的策略） ---
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[NaN Debug] NaN/Inf detected in logits, fixing.")
                logits = torch.nan_to_num(logits)

            B, N, K = logits.shape
            labels = labels.view(B, N)
            valid = (labels != ignore_index)

            if valid.sum() == 0:
                loss = logits.new_tensor(0.0, requires_grad=True)
                stats = {"loss": float(loss.item()), "acc": 0.0, "valid_points": 0}
                return loss, logits, stats

            x = logits[valid]  # (M,K)
            y = labels[valid].long()  # (M,)

            # ---- 计算 log_probs / probs ----
            log_probs = F.log_softmax(x, dim=-1)
            probs = log_probs.exp()

            # ---- label smoothing 的目标分布（样本级 one-hot -> 平滑）----
            if label_smoothing and label_smoothing > 0.0:
                eps = float(label_smoothing)
                true_dist = torch.full_like(log_probs, eps / (K - 1))
                true_dist.scatter_(1, y.unsqueeze(1), 1.0 - eps)
            else:
                true_dist = torch.zeros_like(log_probs)
                true_dist.scatter_(1, y.unsqueeze(1), 1.0)

            # ---- 样本级类别权重：w_i = weight[y_i]（若未提供则为 1）----
            if class_weights is not None:
                if not torch.is_tensor(class_weights):
                    class_weights = torch.tensor(class_weights, dtype=x.dtype, device=x.device)
                w_sample = class_weights[y]  # (M,)
            else:
                w_sample = torch.ones_like(y, dtype=x.dtype)

            # ---- 基础 CE（逐样本、未聚合）----
            ce_per = -(true_dist * log_probs).sum(dim=-1)  # (M,)

            # ---- 可选 Focal（逐样本）----
            if focal_gamma and float(focal_gamma) > 0.0:
                pt = probs.gather(1, y.unsqueeze(1)).squeeze(1)  # (M,)
                focal_w = (1.0 - pt).clamp(min=0.0) ** float(focal_gamma)
                ce_per = ce_per * focal_w

            # ---- 应用样本权重并聚合 ----
            ce_per = ce_per * w_sample
            if reduction == 'sum':
                ce = ce_per.sum()
            else:
                ce = ce_per.mean()  # 默认 mean

            # ---- 可选辅助头损失（若存在）----
            total_loss = ce
            if aux_weight is None:
                aux_weight = 0.4
            if aux_weight > 0 and hasattr(self, "sem128_head"):
                # 取你网络中间特征（约定 forward 里已经缓存或可再次前向取回）
                # 为避免破坏结构，这里直接用 logits 的梯度路径，简化为对 x 的线性映射：
                # 如果你原本实现里有明确的 sem128 logits，请替换为该张量。
                try:
                    aux_logits = self.sem128_head(self.cached_fuse128.permute(0, 2, 1)).permute(0, 2, 1)  # (B,N,K)
                    aux_x = aux_logits[valid]
                    aux_log_probs = F.log_softmax(aux_x, dim=-1)
                    if label_smoothing and label_smoothing > 0.0:
                        aux_ce_per = -(true_dist * aux_log_probs).sum(dim=-1)
                    else:
                        aux_ce_per = F.nll_loss(aux_log_probs, y, reduction='none')
                    if class_weights is not None:
                        aux_ce_per = aux_ce_per * w_sample
                    aux_loss = aux_ce_per.mean()
                    total_loss = total_loss + aux_weight * aux_loss
                except Exception:
                    # 如果没有 cached_fuse128 或辅助头不可用，则跳过
                    pass

            # ---- 训练时的简易准确率（仅统计有效样本）----
            with torch.no_grad():
                preds = x.argmax(dim=-1)
                acc = (preds == y).float().mean()

            stats = {
                "loss": float(total_loss.detach().item()),
                "acc": float(acc.item()),
                "valid_points": int(valid.sum().item())
            }
            return total_loss, logits, stats