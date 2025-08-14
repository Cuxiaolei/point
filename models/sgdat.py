# sgdat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容导入（允许作为包或脚本）
try:
    from .modules_sgdat import (
        ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion,
        nearest_interpolate, idw_interpolate, SemanticGuidedGate
    )
except Exception:
    from modules_sgdat import (
        ChannelCCC, LinearSpatialGVA, DropPath, DynamicRadiusChannelFusion,
        nearest_interpolate, idw_interpolate, SemanticGuidedGate
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

# -----------------utils: fps & indexing----------------


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
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    return: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# -------------------------
# 轻量模块（点形式到 Conv1d 的桥接）
# -------------------------
class SE1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv1d(hidden, channels, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):  # x: [B,C,N]
        s = self.pool(x)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)

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
    def __init__(self,
                 in_dim=9,
                 base_dim=64,
                 num_classes=3,
                 bn_eps=1e-3,
                 bn_momentum=0.01,
                 dropout_p=0.0,
                 use_channel_ccc=True,
                 use_linear_gva=True,
                 use_dynamic_fusion=True,
                 use_semantic_guided_fusion=True,
                 gva_lin_embed=64,
                 dyn_neighbors=16,
                 dyn_tau=0.2,
                 logit_temp=1.0,
                 debug=False):
        super().__init__()
        self.in_dim = in_dim
        self.base_dim = base_dim
        self.num_classes = num_classes
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.dropout_p = dropout_p
        self.use_channel_ccc = use_channel_ccc
        self.use_linear_gva = use_linear_gva
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_semantic_guided_fusion = use_semantic_guided_fusion
        self.gva_lin_embed = gva_lin_embed
        self.dyn_neighbors = dyn_neighbors
        self.dyn_tau = dyn_tau
        self.logit_temp = logit_temp
        self.debug = debug

        # ---------- stem ----------
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim - 3, base_dim, 1, bias=False),
            nn.BatchNorm1d(base_dim, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        # ---------- encoder ----------
        self.use_geom_enhance = True
        self.geom_embed_512 = nn.Sequential(
            nn.Linear(10, base_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim // 2, base_dim // 2, bias=False)
        )
        self.pos512 = nn.Sequential(
            nn.Linear(3, base_dim * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim * 2, base_dim * 2, bias=False)
        )
        self.posN = nn.Sequential(
            nn.Linear(3, base_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(base_dim, base_dim, bias=False)
        )

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
                c_in=base_dim, c_hidden=base_dim, c_out=base_dim,
                k=self.dyn_neighbors, tau=self.dyn_tau
            )
            self.dyn128 = DynamicRadiusChannelFusion(
                c_in=base_dim * 2, c_hidden=base_dim * 2, c_out=base_dim * 2,
                k=self.dyn_neighbors, tau=self.dyn_tau
            )
        else:
            self.dyn512 = None
            self.dyn128 = None

        # 语义引导分支
        if self.use_semantic_guided_fusion:
            self.sem128_head = nn.Sequential(
                nn.Conv1d(base_dim * 2, base_dim * 2, 1, bias=False),
                nn.BatchNorm1d(base_dim * 2, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_dim * 2, num_classes, 1, bias=True)
            )
            self.gate_to_512 = SemanticGuidedGate(num_classes=num_classes, temperature=1.0, use_max_class=True)
            self.gate_to_N = SemanticGuidedGate(num_classes=num_classes, temperature=1.0, use_max_class=True)
        else:
            self.sem128_head = None
            self.gate_to_512 = None
            self.gate_to_N = None

        # 解码与融合
        self.branch_dp1 = DropPath(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.branch_dp2 = DropPath(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self.up1 = nn.Sequential(
            nn.Conv1d(base_dim + base_dim * 2 + base_dim * 2, base_dim * 2, 1, bias=False),
            nn.BatchNorm1d(base_dim * 2, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.up1_refine = ResMLP1D(base_dim * 2, base_dim * 2, bn_eps=bn_eps, bn_momentum=bn_momentum,
                                   use_se=True, dropout=self.dropout_p)
        self.up2 = nn.Sequential(
            nn.Conv1d(base_dim * 2 + base_dim + base_dim, base_dim, 1, bias=False),
            nn.BatchNorm1d(base_dim, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.up2_refine = ResMLP1D(base_dim, base_dim, bn_eps=bn_eps, bn_momentum=bn_momentum,
                                   use_se=True, dropout=self.dropout_p)

        if self.use_linear_gva:
            self.gva_512 = LinearSpatialGVA(base_dim * 2)
            self.gva_N = LinearSpatialGVA(base_dim)
        else:
            self.gva_512 = nn.Identity()
            self.gva_N = nn.Identity()

        self.head = nn.Sequential(
            nn.Conv1d(base_dim, base_dim, 1, bias=False),
            nn.BatchNorm1d(base_dim, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_dim, num_classes, 1, bias=True)
        )

        # ===== add: 对齐 gva_N 输入通道 (128 -> 64) =====
        self.align_gva_in = nn.Linear(base_dim * 2, base_dim, bias=False)

        self.apply(self._init_stable)

    # ===== add: 简单的 NaN/Inf 清理工具 =====
    def _safe_clean(self, x, clip_val=1e4):
        # 将 NaN 置 0，将 Inf 截断
        x = torch.nan_to_num(x, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return x
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

    def _apply_channel_first_module(self, x_bnC, module_cf):
        # 输入 [B,N,C] -> [B,C,N] -> 模块 -> [B,N,C]
        x_cf = x_bnC.permute(0, 2, 1).contiguous()
        y_cf = module_cf(x_cf) if not isinstance(module_cf, nn.Identity) else x_cf
        y_bnC = y_cf.permute(0, 2, 1).contiguous()
        return y_bnC

    def _apply_channel_first_module_with_pos(self, feat_bnC, xyz_bn3, module_cf):
        # DynamicRadiusChannelFusion 这类模块需要 [B,C,N] 和 [B,3,N]
        f_cf = feat_bnC.permute(0, 2, 1).contiguous()
        p_cf = xyz_bn3.permute(0, 2, 1).contiguous()
        out_cf = module_cf(f_cf, p_cf)
        out_bnC = out_cf.permute(0, 2, 1).contiguous()
        return out_bnC

    def _check_nan(self, name, tensor):
        """检测张量中的 NaN/Inf 并打印调试信息"""
        if tensor is None:
            print(f"[DEBUG] {name}: None")
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"[ALERT] {name} has NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
            print(f"[ALERT] Sample from {name}: {tensor.view(-1)[:10]}")
        else:
            print(f"[OK] {name}: no NaN/Inf, shape={tensor.shape}")

    def forward(self, points, return_aux=False):
        """
        points: [B, N, in_dim], in_dim=9 -> xyz(3) + rgb(3) + normal(3)
        """
        B, N, C = points.shape
        xyz = points[:, :, :3].contiguous()
        feat0 = points[:, :, 3:].contiguous()  # (B,N,6)
        self._check_nan(points, "input_points")

        # 归一化坐标（作为位置编码输入）
        center = xyz.mean(dim=1, keepdim=True)
        xyz_normed = xyz - center
        scale = torch.clamp(xyz_normed.norm(dim=-1).max(dim=1)[0].view(B, 1, 1), min=1e-6)
        xyz_normed = xyz_normed / scale
        self._check_nan(xyz_normed, "xyz_normed")

        # stem
        feat0_cf = feat0.permute(0, 2, 1).contiguous()  # [B,6,N]
        stem = self.stem(feat0_cf)                      # [B,64,N]
        feat0_64 = stem.permute(0, 2, 1).contiguous()   # [B,N,64]
        self._check_nan(feat0_64, "feat0_64")

        # 下采样到 512
        xyz_512, feat0_512, idx_512 = self._sample_and_gather(xyz_normed, feat0_64, npoint=512)
        self._check_nan(feat0_512, "feat0_512")

        # 取下采样后的 rgb 与 normal
        rgb_512 = batched_index_points(points[:, :, 3:6], idx_512)  # (B,512,3)
        norm_512 = batched_index_points(points[:, :, 6:9], idx_512)  # (B,512,3)

        # 颜色差和法向差（相对均值）
        rgb_diff = rgb_512 - rgb_512.mean(dim=1, keepdim=True)  # (B,512,3)
        norm_diff = norm_512 - norm_512.mean(dim=1, keepdim=True)  # (B,512,3)

        # 颜色相似度（cosine）
        rgb_cos = F.cosine_similarity(rgb_512, rgb_512.mean(dim=1, keepdim=True), dim=-1, eps=1e-6).unsqueeze(-1)

        # 法向相似度（cosine）
        norm_cos = F.cosine_similarity(norm_512, norm_512.mean(dim=1, keepdim=True), dim=-1, eps=1e-6).unsqueeze(-1)

        geom_512 = torch.cat([
            xyz_512,  # 3
            torch.norm(xyz_512, dim=-1, keepdim=True),  # 1
            torch.mean(torch.abs(xyz_512 - xyz_512.mean(dim=1, keepdim=True)), dim=-1, keepdim=True),  # 1
            torch.norm(rgb_diff, dim=-1, keepdim=True),  # 1
            torch.norm(norm_diff, dim=-1, keepdim=True),  # 1
            rgb_cos,  # 1
            norm_cos,  # 1
            torch.zeros_like(xyz_512[..., :1])  # 1
        ], dim=-1)  # [B,512,10]
        self._check_nan(geom_512, "geom_512")

        if self.use_geom_enhance:
            geom_emb_512 = self.geom_embed_512(geom_512)                   # (B,512,base_dim//2)
            enc_512_in = torch.cat([xyz_512, feat0_512, geom_emb_512], dim=-1)
        else:
            enc_512_in = torch.cat([xyz_512, feat0_512], dim=-1)
        feat_512 = self.enc512(enc_512_in)                                 # (B,512,64)
        self._check_nan(feat_512, "feat_512_after_enc")

        # —— 动态邻域-通道融合 @512
        if self.use_dynamic_fusion and self.dyn512 is not None:
            feat_512 = feat_512 + self._apply_channel_first_module_with_pos(feat_512, xyz_512, self.dyn512)
            self._check_nan(feat_512, "feat_512_after_dyn")

        # 通道注意力 @512
        feat_512 = self._apply_channel_first_module(feat_512, self.ccc_512)  # (B,512,64)
        self._check_nan(feat_512, "feat_512_after_ccc")
        xyz_128, feat_512_128, idx_128 = self._sample_and_gather(xyz_512, feat_512, npoint=128)
        enc_128_in = torch.cat([xyz_128, feat_512_128], dim=-1)
        feat_128 = self.enc128(enc_128_in)
        self._check_nan("enc128_out", feat_128)

        if self.use_dynamic_fusion and self.dyn128 is not None:
            feat_128 = feat_128 + self._apply_channel_first_module_with_pos(feat_128, xyz_128, self.dyn128)
        self._check_nan("dyn128_out", feat_128)

        feat_128 = self._apply_channel_first_module(feat_128, self.ccc_128)

        # 语义引导
        gate512 = gateN = None
        sem128_logits = None
        if self.use_semantic_guided_fusion and self.sem128_head is not None:
            sem128_logits = self.sem128_head(feat_128.permute(0, 2, 1).contiguous())
            gate512 = self.gate_to_512(sem128_logits, source_pos=xyz_128.permute(0, 2, 1).contiguous(),
                                       target_pos=xyz_512.permute(0, 2, 1).contiguous())
            gateN = self.gate_to_N(sem128_logits, source_pos=xyz_128.permute(0, 2, 1).contiguous(),
                                   target_pos=xyz_normed.permute(0, 2, 1).contiguous())

        # Decoder: 128 -> 512
        up128_to_512_cf = idw_interpolate(
            target_pos=xyz_512.permute(0, 2, 1).contiguous(),
            source_pos=xyz_128.permute(0, 2, 1).contiguous(),
            source_feat=feat_128.permute(0, 2, 1).contiguous(),
            k=3, p=2.0
        )
        up128_to_512 = up128_to_512_cf.permute(0, 2, 1).contiguous()

        if gate512 is not None:
            g512 = gate512
            if g512.shape[1] == 1 and g512.shape[2] == up128_to_512.shape[1]:
                g512 = g512.permute(0, 2, 1).contiguous()
            up128_to_512 = up128_to_512 * g512.expand_as(up128_to_512)

        pos512 = self.pos512(xyz_512)
        up128_to_512 = self.branch_dp1(up128_to_512)
        pos512 = self.branch_dp1(pos512)

        fuse_512 = torch.cat([feat_512, up128_to_512, pos512], dim=-1)
        fuse_512 = self.up1(fuse_512.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        fuse_512 = self.up1_refine(fuse_512)
        fuse_512 = self._apply_channel_first_module(fuse_512, self.gva_512)

        # Decoder: 512 -> N
        up512_to_N_cf = idw_interpolate(
            target_pos=xyz_normed.permute(0, 2, 1).contiguous(),
            source_pos=xyz_512.permute(0, 2, 1).contiguous(),
            source_feat=fuse_512.permute(0, 2, 1).contiguous(),
            k=3, p=2.0
        )
        up512_to_N = up512_to_N_cf.permute(0, 2, 1).contiguous()
        self._check_nan("up512_to_N_before_gva", up512_to_N)

        if gateN is not None:
            gN = gateN
            if gN.shape[1] == 1 and gN.shape[2] == up512_to_N.shape[1]:
                gN = gN.permute(0, 2, 1).contiguous()
            up512_to_N = up512_to_N * gN.expand_as(up512_to_N)

        posN = self.posN(xyz_normed)

        if self.use_linear_gva:
            up512_to_N = self._safe_clean(up512_to_N)
            self._check_nan("up512_to_N_after_clean", up512_to_N)
            up512_to_N = self._apply_channel_first_module(up512_to_N, self.gva_N)

        fuse_N = torch.cat([up512_to_N, feat0, posN], dim=-1)
        fuse_N = self.up2(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        fuse_N = self.up2_refine(fuse_N)

        logits = self.head(fuse_N.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        if self.logit_temp and self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        if return_aux and (sem128_logits is not None):
            aux = {
                "sem128_logits": sem128_logits,
                "idx_128": idx_128
            }
            return logits, aux

        return logits

    # ---------------- loss（保持你的完整实现） ----------------
    def get_loss(
            self,
            points,
            labels,
            class_weights=None,
            ignore_index=-1,
            aux_weight: float = 0.2,  # 语义引导辅助监督的权重
            reduction='mean',
            label_smoothing: float = 0.0,
            focal_gamma: float = 0.0,
            **kwargs
    ):
        # 前向时请求辅助输出
        out = self.forward(points, return_aux=True)
        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits, aux = out, None

        B, N, K = logits.shape

        # 有效掩码
        labels = labels.long()
        valid_mask = labels.ne(ignore_index)
        logits_valid = logits[valid_mask]
        labels_valid = labels[valid_mask]

        # class weights
        weight = None
        if class_weights is not None:
            weight = class_weights.to(logits.device, dtype=logits.dtype)

        # -------- 主损失：CE + 可选 smoothing/focal --------
        if label_smoothing is None or label_smoothing <= 0.0:
            if focal_gamma is None or focal_gamma <= 0.0:
                ce = F.cross_entropy(logits_valid, labels_valid, reduction='none', weight=weight)
            else:
                # Focal CE
                log_probs = F.log_softmax(logits_valid, dim=-1)
                probs = log_probs.exp()
                ce_raw = F.nll_loss(log_probs, labels_valid, reduction='none', weight=weight)
                pt = probs.gather(1, labels_valid.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
                focal_weight = (1.0 - pt) ** float(focal_gamma)
                ce = focal_weight * ce_raw
        else:
            eps = float(label_smoothing)
            log_probs = F.log_softmax(logits_valid, dim=-1)
            probs = log_probs.exp()
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.scatter_(1, labels_valid.unsqueeze(1), 1.0)

            ce = -(true_dist * log_probs).sum(dim=-1)

            if weight is not None:
                w = weight[labels_valid]
                ce = ce * w

            if focal_gamma is not None and focal_gamma > 0.0:
                gamma = float(focal_gamma)
                pt = (true_dist * probs).sum(dim=-1).clamp(min=1e-6, max=1.0)
                if torch.isnan(pt).any() or torch.isinf(pt).any():
                    print("[NaN Debug] NaN/Inf detected in pt, fixing...")
                    pt = torch.nan_to_num(pt)
                focal_weight = (1.0 - pt) ** gamma
                ce = focal_weight * ce

        if reduction == 'mean':
            loss_main = ce.mean()
        elif reduction == 'sum':
            loss_main = ce.sum()
        else:
            loss_main = ce

        # ===== 语义门控的辅助监督（CE@128） =====
        aux_loss = logits.new_tensor(0.0)
        if aux is not None and self.use_semantic_guided_fusion and (aux_weight is not None) and aux_weight > 0.0:
            sem128_logits = aux["sem128_logits"]  # (B,K,128) channel-first
            idx_128 = aux["idx_128"]  # (B,128)

            # 下采样标签：用 idx_128 从原 N 标签挑选到 128
            labels_128 = batched_index_points(labels.unsqueeze(-1), idx_128).squeeze(-1)  # (B,128)
            valid_128 = labels_128.ne(ignore_index)

            # 只对有效位置监督
            sem_logits_flat = sem128_logits.permute(0, 2, 1).contiguous().view(-1, K)  # (B*128,K)
            labels_128_flat = labels_128.reshape(-1)  # (B*128,)
            valid_128_flat = valid_128.reshape(-1)

            sem_logits_valid = sem_logits_flat[valid_128_flat]
            labels_128_valid = labels_128_flat[valid_128_flat]

            if sem_logits_valid.numel() > 0:
                aux_ce = F.cross_entropy(sem_logits_valid, labels_128_valid, reduction='mean', weight=weight)
                aux_loss = aux_ce * float(aux_weight)

        loss = loss_main + aux_loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("[NaN Debug] NaN/Inf detected in final loss, fixing...")
            loss = torch.nan_to_num(loss)

        with torch.no_grad():
            pred = logits_valid.argmax(dim=-1)
            correct = (pred == labels_valid).sum()
            acc = correct.float() / labels_valid.numel()

        stats = {
            "loss": float(loss.detach().item()),
            "acc": float(acc.detach().item()),
            "valid_points": int(valid_mask.sum().item())
        }
        return loss, logits, stats
