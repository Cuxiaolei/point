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
# 新增：可配置的数值稳定参数（从 config 中读取，无则采用这里的默认）
# =========================
DEFAULTS = {
    "CLIP_NORM": 2.0,
    "DET_COLLECT_LOGITS_NAN_ONCE": True,
    "SKIP_STEP_LOSS_THRESH": 1e6,  # loss 超过阈值跳过该 step
    "DETECT_ANOMALY": False,       # 开启 autograd 异常检测（调试时用，训练建议关）
    "POINT_DROPOUT_ENABLE": False,
    "POINT_DROPOUT_RATE": 0.1,
    "POINTCUTMIX_ENABLE": False,
    "POINTCUTMIX_PROB": 0.1,
    "POINTCUTMIX_RATIO": 0.2,
    "ENABLE_SOFT_DICE": False,
    "SOFT_DICE_EPS": 1e-6,
    "SOFT_DICE_WEIGHT": 0.5,
    "ENABLE_TEMP_SCALING": False,
    "TEMP_FACTOR": 1.5,
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


class Trainer:
    """
    训练器（增强版）
    - 训练/验证均计算 mIoU（训练新增）
    - 训练结束后输出每轮指标汇总到日志（train.log）
    - 其余：AMP、EMA、CutMix、点丢弃、早停、TensorBoard 等保持不变
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
        # 避免重复添加 handler
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

        # 自定义 collate（保证 shape 固定或不固定，依据 LIMIT_POINTS）
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
                # 不限制点数：pad 到本 batch 最大长度
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

    def _compute_class_weights(self):
        """统计训练集中每类样本数并返回归一化权重张量"""
        counts = np.zeros(self.config.NUM_CLASSES, dtype=np.int64)
        # 与你原来一致的统计方式（scene_list/segment20.npy）
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
        return tensor_weights

    @torch.no_grad()
    def _sanitize_batch(self, points, labels):
        """
        根源级批次清理：处理 NaN/Inf/极值 & 标签越界
        - points: (B,N,C) -> nan_to_num，clamp 到合理范围（避免坐标爆炸）
        - labels: (B,N)   -> 合法集合 { -1, 0..K-1 }，非法置 -1
        """
        # points
        torch.nan_to_num_(points, nan=0.0, posinf=1e6, neginf=-1e6)
        # 可选对坐标/特征做全局裁剪（若你的数据范围更大可调大些）
        points.clamp_(min=-1e6, max=1e6)

        # labels
        K = self.config.NUM_CLASSES
        valid_mask = (labels == -1) | ((labels >= 0) & (labels < K))
        labels = torch.where(valid_mask, labels, torch.full_like(labels, -1))
        return points, labels

    def _apply_temp(self, logits):
        """可选温度缩放：降低过度自信，提高稳定性/兼容轻量注意力输出"""
        if _cfg_val(self.config, "ENABLE_TEMP_SCALING"):
            T = float(max(1e-6, _cfg_val(self.config, "TEMP_FACTOR")))
            return logits / T
        return logits

    def _soft_dice_loss(self, logits, labels, num_classes, ignore_index=-1, eps=1e-6):
        """可选 soft dice，点级 one-vs-rest，忽略 label=-1 的 pad。返回标量。"""
        # logits -> probs
        probs = torch.softmax(logits, dim=-1)  # (B,N,K)
        B, N, K = probs.shape
        mask = (labels != ignore_index).float()  # (B,N)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        dice_sum = 0.0
        valid_c = 0
        # 展平到 (B*N, K) 同时带掩码
        probs = probs.reshape(B * N, K)
        labels = labels.reshape(B * N)
        mask = mask.reshape(B * N)

        for c in range(num_classes):
            target = (labels == c).float() * mask
            pred   = probs[:, c] * mask
            inter = (pred * target).sum()
            denom = pred.sum() + target.sum() + eps
            dice_c = (2 * inter + eps) / denom
            dice_sum += (1.0 - dice_c)  # 损失 = 1 - dice
            valid_c += 1
        if valid_c == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return dice_sum / valid_c

    def _fallback_loss(self, logits, labels,
                       class_weights=None, ignore_index=-1,
                       label_smoothing=0.0, focal_gamma=0.0):
        """
        当模型暂时没有 get_loss 时的稳健回退 CE/Focal + （可选）Soft-Dice 组合
        """
        # 温度缩放（与验证端保持一致）
        logits = self._apply_temp(logits)

        # 展平
        B, N, K = logits.shape
        x = logits.reshape(B * N, K)
        y = labels.reshape(B * N)

        # 先做稳定 cross_entropy（带 label smoothing）
        ce = torch.nn.functional.cross_entropy(
            x, y,
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

        if focal_gamma and focal_gamma > 0.0:
            # 简易 focal：在有效样本上基于预测概率重新加权
            with torch.no_grad():
                pt = torch.softmax(x, dim=-1).gather(1, torch.clamp(y.unsqueeze(1), 0, K-1)).squeeze(1)
                mask = (y != ignore_index)
                pt = torch.where(mask, pt, torch.ones_like(pt))
                focal_w = (1 - pt) ** focal_gamma
            ce = (ce * focal_w.mean())  # 标量缩放（避免与ignore_index冲突）

        loss = ce

        # 组合 Soft-Dice
        if _cfg_val(self.config, "ENABLE_SOFT_DICE"):
            dice = self._soft_dice_loss(
                logits.view(B, N, K), labels.view(B, N),
                num_classes=K,
                ignore_index=ignore_index,
                eps=_cfg_val(self.config, "SOFT_DICE_EPS"),
            )
            w = float(_cfg_val(self.config, "SOFT_DICE_WEIGHT"))
            loss = (1.0 - w) * loss + w * dice

        return loss

    def _miou_from_labels_preds(self, labels_list, preds_list, num_classes):
        """从累积的标签/预测计算 per-class IoU 与 mIoU"""
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

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        # 训练集也累计 IoU
        train_lbls_all = []
        train_preds_all = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.MAX_EPOCHS}")
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)   # (B,N,C)
            labels = labels.to(self.device, non_blocking=True)   # (B,N)

            # 根源清理：NaN/Inf/越界标签
            points, labels = self._sanitize_batch(points, labels)

            # --- 反过拟合策略：点丢弃 ---
            if _cfg_val(self.config, "POINT_DROPOUT_ENABLE"):
                points, labels = random_point_dropout(
                    points, labels, drop_rate=_cfg_val(self.config, "POINT_DROPOUT_RATE")
                )

            # --- 反过拟合策略：PointCutMix ---
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
                        label_smoothing=(self.config.LABEL_SMOOTH_FACTOR if getattr(self.config, "ENABLE_LABEL_SMOOTH", True) else 0.0),
                        focal_gamma=(self.config.FOCAL_GAMMA if getattr(self.config, "ENABLE_FOCAL_LOSS", True) else 0.0),
                    )
                else:
                    logits = self.model(points)
                    # 温度缩放 + 组合损失
                    loss = self._fallback_loss(
                        logits, labels,
                        class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                        ignore_index=-1,
                        label_smoothing=(self.config.LABEL_SMOOTH_FACTOR if getattr(self.config, "ENABLE_LABEL_SMOOTH", True) else 0.0),
                        focal_gamma=(self.config.FOCAL_GAMMA if getattr(self.config, "ENABLE_FOCAL_LOSS", True) else 0.0)
                    )
                    # 确保 logits 经过温度以便后续指标一致
                    logits = self._apply_temp(logits)

            # 数值检查
            if (not torch.isfinite(loss)) or (loss.item() > _cfg_val(self.config, "SKIP_STEP_LOSS_THRESH")):
                save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"bad_loss_epoch{epoch}_batch{batch_idx}.pth")
                torch.save(self.model.state_dict(), save_path)
                self.logger.warning(f"[SkipStep] 非法或过大损失 loss={loss.item():.4g}，跳过该步。模型已保存到 {save_path}")
                # 跳过优化，但继续训练
                continue

            # 反传 + 梯度裁剪 + step
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                # 检查梯度是否有限
                grads_finite = True
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    self.logger.warning("[SkipStep] 梯度出现 NaN/Inf，跳过该步更新。")
                    # 清梯度并推进 scaler 状态，避免下一步再次调用 unscale_ 报错
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()  # ★ 关键修复：推进 AMP 内部状态
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                # 检查梯度是否有限
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

            # metrics (exclude padded labels -1)
            mask = (labels != -1)
            if mask.sum() == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": "N/A", "有效点": 0})
            else:
                with torch.no_grad():
                    # 有些模型的 get_loss 已返回 logits；保守处理：必要时再前向一次
                    if 'logits' not in locals() or logits is None:
                        logits = self.model(points)
                        logits = self._apply_temp(logits)
                    # 一次性 logits NaN 告警（可关）
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

        # 控制台 & 日志
        msg = f"训练轮次 {epoch}：损失 = {epoch_loss:.4f}, 准确率 = {epoch_acc:.4f}, mIoU = {train_miou:.4f}"
        print(msg)
        self.logger.info(msg)

        return epoch_loss, epoch_acc, train_miou, train_per_class_iou

    def validate(self, epoch):
        # 如果启用 EMA，则用 shadow 权重评估
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

                # 验证也做批次清理，避免偶发脏值影响指标
                points, labels = self._sanitize_batch(points, labels)

                with autocast(device_type='cuda', enabled=self.scaler is not None):
                    if hasattr(self.model, "get_loss"):
                        loss, logits, _ = self.model.get_loss(
                            points, labels,
                            class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                            ignore_index=-1,
                            aux_weight=getattr(self.config, "SEMANTIC_AUX_LOSS_WEIGHT", 0.4),
                            label_smoothing=0.0,
                            focal_gamma=0.0
                        )
                    else:
                        logits = self.model(points)
                        loss = self._fallback_loss(
                            logits, labels,
                            class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                            ignore_index=-1,
                            label_smoothing=0.0,
                            focal_gamma=0.0
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

        # 打印 per-class IoU（仅展示出现过的类为主）
        present = [c for c, v in per_class_iou.items() if v > 0]
        print("[IoU per class]")
        for cls in range(self.config.NUM_CLASSES):
            print(f"  [IoU] class {cls}: {per_class_iou.get(cls, 0.0):.4f}")

        if using_ema:
            self.ema.restore(self.model)

        return val_loss, val_acc, miou, per_class_iou

    def _summarize_history(self):
        """训练完成后，将每轮的 train/val 指标汇总打印并写入日志文件"""
        if len(self.epoch_records) == 0:
            return

        # 表头
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

                # === 注意：这里的 scheduler.step() 在整轮优化器更新之后调用（不会触发 PyTorch 警告）===
                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

                # 预填本轮记录（val 指标先占位 None）
                record = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "train_miou": float(train_miou),
                    "val_loss": None, "val_acc": None, "val_miou": None
                }

                # validate periodically
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
                        # 早停计数
                        if getattr(self.config, "EARLY_STOP_ENABLE", False):
                            self.early_stop_counter += 1
                            if self.early_stop_counter >= getattr(self.config, "EARLY_STOP_PATIENCE", 10):
                                print(f"[EarlyStopping] 验证 mIoU 连续 {self.early_stop_counter} 次未提升，提前停止训练。")
                                self.logger.info(f"[EarlyStopping] 验证 mIoU 连续 {self.early_stop_counter} 次未提升，提前停止训练。")
                                self.epoch_records.append(record)
                                break

                # 记录本轮
                self.epoch_records.append(record)

                # periodic save
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

        # final validate（保持一次最终验证）
        val_loss, val_acc, val_miou, _ = self.validate(self.config.MAX_EPOCHS)
        print(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.logger.info(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.writer.close()

        # 训练结束后输出整表总结
        self._summarize_history()
