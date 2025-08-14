import os
import copy
import math
import random
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
import torch.nn.utils as nn_utils

# =========================
# 可配置的数值稳定 & 训练节奏参数（若 config 中存在同名将覆盖）
# =========================
DEFAULTS = {
    # 数值稳定
    "CLIP_NORM": 2.0,
    "DET_COLLECT_LOGITS_NAN_ONCE": True,
    "SKIP_STEP_LOSS_THRESH": 1e6,
    "DETECT_ANOMALY": False,
    # 数据层/输入层正则
    "POINT_DROPOUT_ENABLE": False,
    "POINT_DROPOUT_RATE": 0.1,
    "POINTCUTMIX_ENABLE": False,
    "POINTCUTMIX_PROB": 0.1,
    "POINTCUTMIX_RATIO": 0.2,
    "FEATURE_DROP_ENABLE": True,
    "FEATURE_DROP_PROB": 0.2,            # 对输入特征维随机失活（不动 xyz）
    "FEATURE_DROP_RANGE": (3, 9),        # 只对 [start,end) 的通道做失活；默认仅 color+normal
    # 组合损失
    "ENABLE_SOFT_DICE": True,
    "SOFT_DICE_EPS": 1e-6,
    "SOFT_DICE_WEIGHT": 0.4,
    "ENABLE_TEMP_SCALING": False,
    "TEMP_FACTOR": 1.5,
    # 训练节奏（warmup）
    "CE_ONLY_EPOCHS": 15,                # 前若干 epoch 只用 CE（关闭 focal & dice & label_smooth）
    "FOCAL_GAMMA": 2.0,                  # 目标最大 gamma
    "FOCAL_WARMUP_EPOCHS": 10,           # 在 CE_ONLY 之后再用这么多 epoch 把 gamma 从 0 线性升到目标
    "LABEL_SMOOTH_FACTOR": 0.1,          # 目标最大 label smoothing
    "LS_WARMUP_EPOCHS": 10,              # 同上，线性升温
    "DICE_WARMUP_EPOCHS": 10,            # 在 CE_ONLY 之后把 dice_weight 从 0 升到 SOFT_DICE_WEIGHT
    # DropPath（若模型支持）
    "DROPPATH_MAX": 0.1,                 # 目标最大 droppath prob（若模型实现了接口）
    # 其他
    "ENABLE_LABEL_SMOOTH": True,
    "ENABLE_FOCAL_LOSS": True,
}

def _cfg_val(cfg, key):
    return getattr(cfg, key, DEFAULTS.get(key))

class ModelEMA:
    """指数滑动平均（不改变原模型结构）"""
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = device
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.detach().clone()

    def apply_shadow(self, model):
        """保存当前权重并替换为 shadow，用于评估"""
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.detach().clone()
            param.data = self.shadow[name].detach().clone()

    def restore(self, model):
        """恢复训练权重"""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name]
        self.backup = {}

# ----------------------
# 批内 CutMix & 点丢弃
# ----------------------
def point_cutmix(points, labels, prob=0.1, ratio=0.2):
    """
    points: (B, N, C), labels: (B, N)
    仅对 label!=-1 的有效点随机抽子集，在 batch 内两两互换
    """
    if prob <= 0 or ratio <= 0:
        return points, labels
    B, N, C = points.shape
    if B < 2:
        return points, labels

    if random.random() > prob:
        return points, labels

    device = points.device
    perm = torch.randperm(B, device=device)
    pts_new = points.clone()
    lbl_new = labels.clone()

    for b in range(B):
        j = perm[b].item()
        if j == b:
            continue
        valid_b = (labels[b] != -1)
        valid_j = (labels[j] != -1)
        nb = valid_b.sum().item()
        nj = valid_j.sum().item()
        if nb == 0 or nj == 0:
            continue

        k = max(1, int(min(nb, nj) * ratio))
        idx_b = torch.nonzero(valid_b, as_tuple=False).squeeze(1)
        idx_j = torch.nonzero(valid_j, as_tuple=False).squeeze(1)

        pick_b = idx_b[torch.randperm(nb, device=device)[:k]]
        pick_j = idx_j[torch.randperm(nj, device=device)[:k]]

        # 互换点与标签
        pts_new[b, pick_b] = points[j, pick_j]
        lbl_new[b, pick_b] = labels[j, pick_j]

    return pts_new, lbl_new

def random_point_dropout(points, labels, drop_rate=0.1):
    """
    对有效点按比例置为无效（label=-1），并将坐标/属性清零
    points: (B, N, C), labels: (B, N)
    """
    if drop_rate <= 0:
        return points, labels
    B, N, C = points.shape
    device = points.device
    pts = points.clone()
    lbl = labels.clone()
    for b in range(B):
        valid = (labels[b] != -1)
        nv = valid.sum().item()
        if nv == 0:
            continue
        k = int(nv * drop_rate)
        if k <= 0:
            continue
        vidx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        pick = vidx[torch.randperm(nv, device=device)[:k]]
        pts[b, pick] = 0.0
        lbl[b, pick] = -1
    return pts, lbl

def random_feature_drop(points, prob=0.2, start=3, end=9):
    """
    随机对输入特征通道置零（不动坐标 xyz）。针对 (B,N,C)。
    start/end 为半开区间 [start, end)，缺省为 color(3)~normal(9)。
    """
    if prob <= 0:
        return points
    B, N, C = points.shape
    if start >= end or start >= C:
        return points
    end = min(end, C)
    pts = points.clone()
    device = pts.device
    # 对每个样本独立抽通道 mask
    for b in range(B):
        mask = torch.ones(C, device=device, dtype=torch.bool)
        for ch in range(start, end):
            if random.random() < prob:
                mask[ch] = False
        # 至少保留一个通道
        if not mask[start:end].any():
            keep_idx = random.randint(start, end - 1)
            mask[keep_idx] = True
        pts[b, :, ~mask] = 0.0
    return pts

class Trainer:
    """
    训练器（增强版）
    - 训练/验证均计算 mIoU（训练新增）
    - 线性 warmup：CE -> 打开 Focal/Dice/LabelSmooth
    - 输入特征随机失活（FeatureDrop）
    - CutMix/点丢弃、EMA、per-class IoU、数值稳定策略
    """

    def __init__(self, model, config, train_dataset, val_dataset, class_weights=None):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.DEVICE

        # ---- 日志器 ----
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(config.LOG_DIR, "train.log"), encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        self.epoch_records = []  # 每轮汇总

        # class weights（来自外部或内部统计）
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.detach().clone().to(self.device, dtype=torch.float32)
            else:
                self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            self.class_weights = None

        # AMP scaler（在 cuda 时启用）
        self.scaler = GradScaler() if self.device.startswith('cuda') else None

        # 最大点数来自 dataset（默认回退）
        self.max_points = getattr(train_dataset, 'max_points', 20000)
        if getattr(config, "LIMIT_POINTS", False):
            print(f"训练器初始化：限制最大点数 {self.max_points}")

        # 自定义 collate（pad -1 到 batch 内最大长度或固定长度）
        def custom_collate(batch):
            points_list = [item[0] for item in batch]  # (N, C)
            labels_list = [item[1] for item in batch]  # (N,)
            if getattr(self.config, "LIMIT_POINTS", False):
                padded_points = []
                padded_labels = []
                for p, l in zip(points_list, labels_list):
                    if p.shape[0] > self.max_points:
                        p = p[:self.max_points]
                        l = l[:self.max_points]
                    elif p.shape[0] < self.max_points:
                        pad_size = self.max_points - p.shape[0]
                        p = torch.cat([p, torch.zeros(pad_size, p.shape[1], device=p.device)], dim=0)
                        l = torch.cat([l, torch.full((pad_size,), -1, device=l.device, dtype=l.dtype)], dim=0)
                    padded_points.append(p)
                    padded_labels.append(l)
                return torch.stack(padded_points), torch.stack(padded_labels)
            else:
                maxN = max(p.shape[0] for p in points_list)
                padded_points = []
                padded_labels = []
                for p, l in zip(points_list, labels_list):
                    if p.shape[0] < maxN:
                        pad_size = maxN - p.shape[0]
                        p = torch.cat([p, torch.zeros(pad_size, p.shape[1], device=p.device)], dim=0)
                        l = torch.cat([l, torch.full((pad_size,), -1, device=l.device, dtype=l.dtype)], dim=0)
                    padded_points.append(p)
                    padded_labels.append(l)
                return torch.stack(padded_points), torch.stack(padded_labels)

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=getattr(config, 'NUM_WORKERS', 4),
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=getattr(config, 'NUM_WORKERS', 4),
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=False
        )

        # optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.MAX_EPOCHS,
            eta_min=1e-5
        )

        # grad clipping threshold（可在 config 中覆写）
        self.clip_norm = _cfg_val(config, 'CLIP_NORM')

        # 日志与保存目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        self.writer = SummaryWriter(config.LOG_DIR)

        # 记录最佳 val mIoU
        self.best_val_miou = 0.0

        # 计算/设定 class weights（若外部未给）
        if self.class_weights is None:
            self.class_weights = self._compute_class_weights()
        print(f"[Trainer] class_weights: {self.class_weights}")

        # EMA
        self.ema = ModelEMA(self.model, decay=config.EMA_DECAY) if getattr(config, "EMA_ENABLE", False) else None

        # Early Stopping
        self.early_stop_counter = 0

        # 仅在需要时打开异常检测
        if _cfg_val(config, "DETECT_ANOMALY"):
            torch.autograd.set_detect_anomaly(True)

        # 打印配置
        if hasattr(config, "print_config"):
            config.print_config()

        # 每轮只提示一次 logits NaN 的标志
        self._logits_nan_warned = False

        # 统计类样本量（可用于日志）
        self.class_counts = getattr(self, "class_counts", None)

    # ---------- 内部工具 ----------
    def _compute_class_weights(self):
        counts = np.zeros(self.config.NUM_CLASSES, dtype=np.int64)
        for scene in getattr(self.train_dataset, "scene_list", []):
            seg_path = os.path.join(scene, 'segment20.npy')
            if os.path.exists(seg_path):
                arr = np.load(seg_path).flatten()
                arr = arr[arr != -1]
                if arr.size > 0:
                    binc = np.bincount(arr, minlength=self.config.NUM_CLASSES)
                    counts += binc
        print("[Trainer] class counts:", counts)
        inv = 1.0 / (counts + 1e-6)
        weights = inv / inv.sum() * float(self.config.NUM_CLASSES)
        weights = np.where(np.isfinite(weights), weights, 1.0)
        tensor_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # 保存到成员变量，便于外部查看
        self.class_counts = counts
        return tensor_weights

    @torch.no_grad()
    def _sanitize_batch(self, points, labels):
        """
        根源级批次清理：处理 NaN/Inf/极值 & 标签越界
        - points: (B,N,C) -> nan_to_num，clamp 到合理范围
        - labels: (B,N)   -> 合法集合 { -1, 0..K-1 }，非法置 -1
        """
        torch.nan_to_num_(points, nan=0.0, posinf=1e6, neginf=-1e6)
        points.clamp_(min=-1e6, max=1e6)

        K = self.config.NUM_CLASSES
        valid_mask = (labels == -1) | ((labels >= 0) & (labels < K))
        labels = torch.where(valid_mask, labels, torch.full_like(labels, -1))
        return points, labels

    def _apply_temp(self, logits):
        """可选温度缩放"""
        if _cfg_val(self.config, "ENABLE_TEMP_SCALING"):
            T = float(max(1e-6, _cfg_val(self.config, "TEMP_FACTOR")))
            return logits / T
        return logits

    def _soft_dice_loss(self, logits, labels, num_classes, ignore_index=-1, eps=1e-6):
        """可选 soft dice，点级 one-vs-rest，忽略 label=-1 的 pad。返回标量。"""
        probs = torch.softmax(logits, dim=-1)  # (B,N,K)
        B, N, K = probs.shape
        mask = (labels != ignore_index).float()  # (B,N)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        dice_sum = 0.0
        valid_c = 0
        probs = probs.reshape(B * N, K)
        labels = labels.reshape(B * N)
        mask = mask.reshape(B * N)

        for c in range(num_classes):
            target = (labels == c).float() * mask
            pred   = probs[:, c] * mask
            inter = (pred * target).sum()
            denom = pred.sum() + target.sum() + eps
            dice_c = (2 * inter + eps) / denom
            dice_sum += (1.0 - dice_c)
            valid_c += 1
        if valid_c == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return dice_sum / valid_c

    def _fallback_loss(self, logits, labels,
                       class_weights=None, ignore_index=-1,
                       label_smoothing=0.0, focal_gamma=0.0,
                       dice_weight=0.0):
        """
        稳健回退 CE/Focal + （可选）Soft-Dice 组合
        """
        logits = self._apply_temp(logits)

        B, N, K = logits.shape
        x = logits.reshape(B * N, K)
        y = labels.reshape(B * N)

        ce = torch.nn.functional.cross_entropy(
            x, y,
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        # focal（以均值缩放避免与 ignore_index 冲突）
        if focal_gamma and focal_gamma > 0.0:
            with torch.no_grad():
                pt = torch.softmax(x, dim=-1).gather(1, torch.clamp(y.unsqueeze(1), 0, K-1)).squeeze(1)
                mask = (y != ignore_index)
                pt = torch.where(mask, pt, torch.ones_like(pt))
                focal_w = (1 - pt) ** focal_gamma
            ce = (ce * focal_w.mean())

        loss = ce

        # 组合 Soft-Dice
        if dice_weight and dice_weight > 0.0:
            dice = self._soft_dice_loss(
                logits.view(B, N, K), labels.view(B, N),
                num_classes=K, ignore_index=ignore_index,
                eps=_cfg_val(self.config, "SOFT_DICE_EPS"),
            )
            w = float(dice_weight)
            loss = (1.0 - w) * loss + w * dice

        return loss

    def _miou_from_labels_preds(self, labels_list, preds_list, num_classes):
        if len(labels_list) == 0:
            per_class_iou = {c: 0.0 for c in range(num_classes)}
            return per_class_iou, 0.0

        all_labels = np.concatenate(labels_list) if isinstance(labels_list[0], np.ndarray) else np.concatenate([x for x in labels_list])
        all_preds  = np.concatenate(preds_list)  if isinstance(preds_list[0], np.ndarray)  else np.concatenate([x for x in preds_list])

        conf = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
        per_class_iou = {}
        present = []
        for cls in range(num_classes):
            tp = int(conf[cls, cls])
            fp = int(conf[:, cls].sum()) - tp
            fn = int(conf[cls, :].sum()) - tp
            denom = tp + fp + fn
            if denom > 0:
                per_class_iou[cls] = float(tp / denom)
                present.append(cls)
            else:
                per_class_iou[cls] = 0.0
        miou = float(np.mean([per_class_iou[c] for c in present])) if present else 0.0
        return per_class_iou, miou

    # ---------- 训练节奏（warmup） ----------
    def _compute_stage_params(self, epoch):
        """
        返回当前 epoch 下应使用的 label_smooth、focal_gamma、dice_weight、droppath_prob
        策略：先 CE_ONLY_EPOCHS 只用 CE；随后各项线性 warmup 到目标值。
        """
        ce_only = int(_cfg_val(self.config, "CE_ONLY_EPOCHS"))
        max_gamma = float(_cfg_val(self.config, "FOCAL_GAMMA"))
        gamma_warm = int(_cfg_val(self.config, "FOCAL_WARMUP_EPOCHS"))
        max_ls = float(_cfg_val(self.config, "LABEL_SMOOTH_FACTOR"))
        ls_warm = int(_cfg_val(self.config, "LS_WARMUP_EPOCHS"))
        max_dice = float(_cfg_val(self.config, "SOFT_DICE_WEIGHT")) if _cfg_val(self.config, "ENABLE_SOFT_DICE") else 0.0
        dice_warm = int(_cfg_val(self.config, "DICE_WARMUP_EPOCHS"))
        droppath_max = float(_cfg_val(self.config, "DROPPATH_MAX"))

        if epoch <= ce_only:
            return 0.0, 0.0, 0.0, 0.0

        t = epoch - ce_only

        gamma = 0.0
        if _cfg_val(self.config, "ENABLE_FOCAL_LOSS") and gamma_warm > 0:
            gamma = max_gamma * min(1.0, t / gamma_warm)

        ls = 0.0
        if _cfg_val(self.config, "ENABLE_LABEL_SMOOTH") and ls_warm > 0:
            ls = max_ls * min(1.0, t / ls_warm)

        dice_w = 0.0
        if max_dice > 0 and dice_warm > 0:
            dice_w = max_dice * min(1.0, t / dice_warm)

        # DropPath（若模型支持）
        dp = droppath_max * min(1.0, epoch / max(1, self.config.MAX_EPOCHS))

        return ls, gamma, dice_w, dp

    def _maybe_set_droppath(self, prob):
        """
        若你的模型实现了 set_stochastic_depth_prob(prob) 或有 droppath_prob 属性，则在每个 epoch 动态设置。
        """
        try:
            if hasattr(self.model, "set_stochastic_depth_prob") and callable(getattr(self.model, "set_stochastic_depth_prob")):
                self.model.set_stochastic_depth_prob(float(prob))
            elif hasattr(self.model, "droppath_prob"):
                setattr(self.model, "droppath_prob", float(prob))
        except Exception:
            # 沉默失败：模型可能未实现
            pass

    # ---------- 一个 epoch ----------
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        # 训练集也累计 IoU
        train_lbls_all = []
        train_preds_all = []

        # 计算本 epoch 的阶段参数并下发到模型（如 droppath）
        label_smooth, focal_gamma, dice_weight, droppath_prob = self._compute_stage_params(epoch)
        self._maybe_set_droppath(droppath_prob)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.MAX_EPOCHS}")
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)   # (B,N,C)
            labels = labels.to(self.device, non_blocking=True)   # (B,N)

            # 根源清理：NaN/Inf/越界标签
            points, labels = self._sanitize_batch(points, labels)

            # 输入特征随机失活（不动 xyz）
            if _cfg_val(self.config, "FEATURE_DROP_ENABLE"):
                s, e = _cfg_val(self.config, "FEATURE_DROP_RANGE")
                points = random_feature_drop(points, prob=_cfg_val(self.config, "FEATURE_DROP_PROB"),
                                             start=s, end=e)

            # 点丢弃
            if _cfg_val(self.config, "POINT_DROPOUT_ENABLE"):
                points, labels = random_point_dropout(
                    points, labels, drop_rate=_cfg_val(self.config, "POINT_DROPOUT_RATE")
                )

            # PointCutMix
            if _cfg_val(self.config, "POINTCUTMIX_ENABLE"):
                points, labels = point_cutmix(
                    points, labels,
                    prob=_cfg_val(self.config, "POINTCUTMIX_PROB"),
                    ratio=_cfg_val(self.config, "POINTCUTMIX_RATIO")
                )

            self.optimizer.zero_grad(set_to_none=True)

            # forward + loss (with AMP)
            with autocast(device_type='cuda', enabled=self.scaler is not None):
                if hasattr(self.model, "get_loss"):
                    loss, logits, stats = self.model.get_loss(
                        points, labels,
                        class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                        ignore_index=-1,
                        aux_weight=getattr(self.config, "SEMANTIC_AUX_LOSS_WEIGHT", 0.4),
                        label_smoothing=label_smooth,
                        focal_gamma=focal_gamma,
                        dice_weight=dice_weight,   # ★ 若你的 get_loss 接受这个参数即可直接用；否则回退里会处理
                    )
                else:
                    logits = self.model(points)
                    loss = self._fallback_loss(
                        logits, labels,
                        class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                        ignore_index=-1,
                        label_smoothing=label_smooth,
                        focal_gamma=focal_gamma,
                        dice_weight=dice_weight
                    )
                    logits = self._apply_temp(logits)

            # 数值检查
            if (not torch.isfinite(loss)) or (loss.item() > _cfg_val(self.config, "SKIP_STEP_LOSS_THRESH")):
                save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"bad_loss_epoch{epoch}_batch{batch_idx}.pth")
                torch.save(self.model.state_dict(), save_path)
                self.logger.warning(f"[SkipStep] 非法或过大损失 loss={loss.item():.4g}，跳过该步。模型已保存到 {save_path}")
                continue

            # 反传 + 梯度裁剪 + step
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                grads_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    self.logger.warning("[SkipStep] 梯度出现 NaN/Inf，跳过该步更新。")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                grads_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    self.logger.warning("[SkipStep] 梯度出现 NaN/Inf，跳过该步更新。")
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.step()

            # EMA 更新
            if self.ema is not None:
                self.ema.update(self.model)

            # metrics（排除 pad）
            mask = (labels != -1)
            if mask.sum() == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": "N/A", "有效点": 0})
            else:
                with torch.no_grad():
                    if 'logits' not in locals() or logits is None:
                        logits = self.model(points)
                        logits = self._apply_temp(logits)

                    if (not self._logits_nan_warned) and _cfg_val(self.config, "DET_COLLECT_LOGITS_NAN_ONCE"):
                        if not torch.isfinite(logits).all().item():
                            self._logits_nan_warned = True
                            self.logger.warning("[NaN Debug] 在 logits 中检测到 NaN/Inf。")

                    preds = logits.argmax(dim=-1)
                    correct = (preds[mask] == labels[mask]).sum().item()
                    total_correct += correct
                    total_points += mask.sum().item()
                    # 累计训练 IoU 数据
                    train_lbls_all.append(labels[mask].detach().cpu().numpy().flatten())
                    train_preds_all.append(preds[mask].detach().cpu().numpy().flatten())

                    batch_acc = correct / mask.sum().item()
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{batch_acc:.4f}",
                        "有效点": f"{mask.sum().item()}"
                    })

            total_loss += loss.item()

            # tensorboard（batch级）
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            if mask.sum() > 0:
                self.writer.add_scalar("train/batch_acc", (correct / max(1, mask.sum().item())), global_step)

            # free mem
            del points, labels, loss, logits
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        # epoch 聚合
        epoch_loss = total_loss / max(1, len(self.train_loader))
        epoch_acc = total_correct / max(1, total_points)

        # 训练集 mIoU
        train_per_class_iou, train_miou = self._miou_from_labels_preds(train_lbls_all, train_preds_all, self.config.NUM_CLASSES)

        # TensorBoard（epoch级）
        self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        self.writer.add_scalar("train/epoch_acc", epoch_acc, epoch)
        self.writer.add_scalar("train/mIoU", train_miou, epoch)

        # 记录 warmup 参数方便对齐
        ls, fg, dw, dp = label_smooth, focal_gamma, dice_weight, droppath_prob
        self.writer.add_scalar("train/label_smooth", ls, epoch)
        self.writer.add_scalar("train/focal_gamma", fg, epoch)
        self.writer.add_scalar("train/dice_weight", dw, epoch)
        self.writer.add_scalar("train/droppath_prob", dp, epoch)

        msg = (f"训练轮次 {epoch}：损失 = {epoch_loss:.4f}, 准确率 = {epoch_acc:.4f}, "
               f"mIoU = {train_miou:.4f} | LS={ls:.3f}, FG={fg:.3f}, DICEw={dw:.3f}, DP={dp:.3f}")
        print(msg)
        self.logger.info(msg)

        return epoch_loss, epoch_acc, train_miou, train_per_class_iou

    def validate(self, epoch):
        using_ema = self.ema is not None
        if using_ema:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"验证轮次 {epoch}")
            for batch_idx, (points, labels) in enumerate(pbar):
                points = points.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if points.numel() == 0:
                    pbar.set_postfix({"val_loss": "N/A", "val_acc": "N/A", "有效点": 0})
                    continue

                points, labels = self._sanitize_batch(points, labels)

                with autocast(device_type='cuda', enabled=self.scaler is not None):
                    if hasattr(self.model, "get_loss"):
                        # 验证端：不使用 focal/dice/ls 的 warmup，只评估主损（或你模型内部定义）
                        loss, logits, _ = self.model.get_loss(
                            points, labels,
                            class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                            ignore_index=-1,
                            aux_weight=getattr(self.config, "SEMANTIC_AUX_LOSS_WEIGHT", 0.4),
                            label_smoothing=0.0,
                            focal_gamma=0.0,
                            dice_weight=0.0
                        )
                    else:
                        logits = self.model(points)
                        loss = self._fallback_loss(
                            logits, labels,
                            class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                            ignore_index=-1,
                            label_smoothing=0.0,
                            focal_gamma=0.0,
                            dice_weight=0.0
                        )
                        logits = self._apply_temp(logits)

                preds = logits.argmax(dim=-1)
                mask = (labels != -1)
                if mask.sum() == 0:
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_acc": "N/A", "有效点": 0})
                    continue

                correct = (preds[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_points += mask.sum().item()
                total_loss += loss.item()

                # 收集用于全局 IoU
                batch_labels = labels[mask].cpu().numpy().flatten()
                batch_preds = preds[mask].cpu().numpy().flatten()
                all_labels.append(batch_labels)
                all_preds.append(batch_preds)

                batch_acc = correct / mask.sum().item()
                pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_acc": f"{batch_acc:.4f}",
                    "有效点": f"{mask.sum().item()}"
                })

                # free mem
                del points, labels, loss, logits, preds, mask
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()

        # 计算 mIoU
        per_class_iou, miou = self._miou_from_labels_preds(all_labels, all_preds, self.config.NUM_CLASSES)

        val_loss = total_loss / max(1, len(self.val_loader))
        val_acc = total_correct / max(1, total_points)

        # logging
        self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
        self.writer.add_scalar("val/epoch_acc", val_acc, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)

        msg = f"验证轮次 {epoch}：损失 = {val_loss:.4f}, 准确率 = {val_acc:.4f}, mIoU = {miou:.4f}"
        print(msg)
        self.logger.info(msg)

        # 打印 per-class IoU
        print("[IoU per class]")
        for cls in range(self.config.NUM_CLASSES):
            print(f"  [IoU] class {cls}: {per_class_iou.get(cls, 0.0):.4f}")

        if using_ema:
            self.ema.restore(self.model)

        return val_loss, val_acc, miou, per_class_iou

    def _summarize_history(self):
        if len(self.epoch_records) == 0:
            return
        header = [
            "Epoch",
            "TrainLoss", "TrainAcc", "TrainmIoU",
            "ValLoss", "ValAcc", "ValmIoU"
        ]
        colw = [6, 10, 9, 11, 9, 8, 9]
        def fmt_cell(s, w):
            s = str(s)
            return s + " " * max(0, w - len(s))

        lines = []
        lines.append(" | ".join(fmt_cell(h, w) for h, w in zip(header, colw)))
        lines.append("-" * (sum(colw) + 3 * (len(colw) - 1)))
        for rec in self.epoch_records:
            row = [
                rec["epoch"],
                f"{rec['train_loss']:.4f}",
                f"{rec['train_acc']:.4f}",
                f"{rec['train_miou']:.4f}",
                (f"{rec['val_loss']:.4f}" if rec['val_loss'] is not None else "-"),
                (f"{rec['val_acc']:.4f}" if rec['val_acc'] is not None else "-"),
                (f"{rec['val_miou']:.4f}" if rec['val_miou'] is not None else "-"),
            ]
            lines.append(" | ".join(fmt_cell(s, w) for s, w in zip(row, colw)))

        summary_text = "\n".join(lines)
        print("\n===== 每轮训练/验证指标总结 =====")
        print(summary_text)
        self.logger.info("\n===== 每轮训练/验证指标总结 =====\n" + summary_text)

    def train(self):
        print("开始训练...")
        try:
            for epoch in range(1, self.config.MAX_EPOCHS + 1):
                train_loss, train_acc, train_miou, _ = self.train_epoch(epoch)

                # 调整学习率
                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

                # 记录
                record = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "train_miou": float(train_miou),
                    "val_loss": None, "val_acc": None, "val_miou": None
                }

                # 周期性验证
                if epoch % self.config.EVAL_FREQ == 0:
                    val_loss, val_acc, val_miou, _ = self.validate(epoch)
                    record.update({
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                        "val_miou": float(val_miou)
                    })

                    # 保存最佳
                    if val_miou > self.best_val_miou:
                        self.best_val_miou = val_miou
                        save_path = os.path.join(self.config.MODEL_SAVE_DIR, "best_model.pth")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "miou": val_miou
                        }, save_path)
                        print(f"保存最佳模型到 {save_path}，mIoU = {val_miou:.4f}")
                        self.logger.info(f"保存最佳模型到 {save_path}，mIoU = {val_miou:.4f}")
                        self.early_stop_counter = 0
                    else:
                        if getattr(self.config, "EARLY_STOP_ENABLE", False):
                            self.early_stop_counter += 1
                            if self.early_stop_counter >= getattr(self.config, "EARLY_STOP_PATIENCE", 10):
                                print(f"[EarlyStopping] 验证 mIoU 连续 {self.early_stop_counter} 次未提升，提前停止训练。")
                                self.logger.info(f"[EarlyStopping] 验证 mIoU 连续 {self.early_stop_counter} 次未提升，提前停止训练。")
                                self.epoch_records.append(record)
                                break

                # 记录本轮
                self.epoch_records.append(record)

                # 周期性保存
                if epoch % self.config.SAVE_FREQ == 0:
                    save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"model_epoch_{epoch}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, save_path)
                    print(f"保存模型到 {save_path}")
                    self.logger.info(f"保存模型到 {save_path}")

        except Exception as e:
            error_save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"error_model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), error_save_path)
            print(f"训练中断，已保存当前模型到 {error_save_path}")
            self.logger.exception(f"训练中断，已保存当前模型到 {error_save_path}")
            raise e

        # 最终验证一次
        val_loss, val_acc, val_miou, _ = self.validate(self.config.MAX_EPOCHS)
        print(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.logger.info(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.writer.close()

        # 汇总
        self._summarize_history()
