import os
import copy
import math
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
import torch.nn.utils as nn_utils

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
    - 自动计算 class weights 并传入 model.get_loss
    - 验证阶段收集 preds/labels，打印分布与 per-class IoU（只对存在类求均值）
    - 数值检测（NaN/Inf）并保存异常模型快照
    - AMP + grad clipping（可通过 config 控制）
    - 反过拟合策略：EMA、PointCutMix、随机点丢弃、早停（全由 config 控制）
    """

    def __init__(self, model, config, train_dataset, val_dataset, class_weights=None):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.DEVICE

        # 历史记录（保留你的原结构）
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_miou": [],
            "val_loss": [],
            "val_acc": [],
            "val_miou": []
        }

        # class weights
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
        # 打印提示只在限制开启时输出
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
                # 不限制点数：做成 list，由模型/损失内部按 mask 处理
                # 但为了下游逻辑一致，这里直接 pad 到本 batch 的最大长度
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
        self.clip_norm = getattr(config, 'CLIP_NORM', 2.0)

        # 日志与保存目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.writer = SummaryWriter(config.LOG_DIR)

        # 记录最佳 val mIoU
        self.best_val_miou = 0.0

        # 计算 class weights（缓解类别不平衡）——保持你的逻辑
        self.class_weights = self._compute_class_weights()
        print(f"[Trainer] class_weights: {self.class_weights}")

        # EMA
        self.ema = ModelEMA(self.model, decay=config.EMA_DECAY) if getattr(config, "EMA_ENABLE", False) else None

        # Early Stopping
        self.early_stop_counter = 0

        # 在初始化完成时打印配置
        config.print_config()

    def _compute_class_weights(self):
        """统计训练集中每类样本数并返回归一化权重张量（保持你的做法）"""
        counts = np.zeros(self.config.NUM_CLASSES, dtype=np.int64)
        for scene in self.train_dataset.scene_list:
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

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.MAX_EPOCHS}")
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)   # (B,N,C)
            labels = labels.to(self.device, non_blocking=True)   # (B,N)

            # --- 反过拟合策略：点丢弃 ---
            if getattr(self.config, "POINT_DROPOUT_ENABLE", False):
                points, labels = random_point_dropout(
                    points, labels, drop_rate=getattr(self.config, "POINT_DROPOUT_RATE", 0.1)
                )

            # --- 反过拟合策略：PointCutMix ---
            if getattr(self.config, "POINTCUTMIX_ENABLE", False):
                points, labels = point_cutmix(
                    points, labels,
                    prob=getattr(self.config, "POINTCUTMIX_PROB", 0.1),
                    ratio=getattr(self.config, "POINTCUTMIX_RATIO", 0.2)
                )

            self.optimizer.zero_grad()
            # forward + loss (with AMP)
            with autocast(device_type='cuda', enabled=self.scaler is not None):
                loss, logits, stats = self.model.get_loss(
                    points, labels,
                    class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                    ignore_index=-1,
                    aux_weight=0.4,
                    label_smoothing=0.05,
                    focal_gamma=1.5,
                )

            # numeric check
            if torch.isnan(loss) or torch.isinf(loss):
                save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"bad_loss_epoch{epoch}_batch{batch_idx}.pth")
                torch.save(self.model.state_dict(), save_path)
                raise RuntimeError(f"Encountered bad loss (NaN/Inf) at epoch {epoch} batch {batch_idx}. Model saved to {save_path}")

            # backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

            # EMA 更新
            if self.ema is not None:
                self.ema.update(self.model)

            # metrics (exclude padded labels -1)
            mask = (labels != -1)
            if mask.sum() == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": "N/A", "有效点": 0})
                continue

            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_points += mask.sum().item()
            total_loss += loss.item()

            if batch_idx % max(1, self.config.LOG_FREQ) == 0:
                batch_acc = correct / mask.sum().item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "有效点": f"{mask.sum().item()}"
                })

            # tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            self.writer.add_scalar("train/batch_acc", correct / max(1, mask.sum().item()), global_step)

            # free mem
            del points, labels, loss, logits, preds, mask
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

        epoch_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        epoch_acc = total_correct / total_points if total_points > 0 else 0.0

        self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        self.writer.add_scalar("train/epoch_acc", epoch_acc, epoch)

        print(f"训练轮次 {epoch}：损失 = {epoch_loss:.4f}, 准确率 = {epoch_acc:.4f}")
        return epoch_loss, epoch_acc

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

                with autocast(device_type='cuda', enabled=self.scaler is not None):
                    loss, logits, _ = self.model.get_loss(
                        points, labels,
                        class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                        ignore_index=-1,
                        aux_weight=0.4,
                        label_smoothing=0.0,
                        focal_gamma=0.0
                    )

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

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
        else:
            all_labels = np.array([], dtype=np.int32)
            all_preds = np.array([], dtype=np.int32)

        conf_matrix = np.zeros((self.config.NUM_CLASSES, self.config.NUM_CLASSES), dtype=np.int64)
        if all_labels.size > 0:
            conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(self.config.NUM_CLASSES))

        per_class_iou = {}
        present = []
        for cls in range(self.config.NUM_CLASSES):
            tp = int(conf_matrix[cls, cls])
            fp = int(conf_matrix[:, cls].sum()) - tp
            fn = int(conf_matrix[cls, :].sum()) - tp
            denom = tp + fp + fn
            if denom > 0:
                iou = tp / denom
                per_class_iou[cls] = float(iou)
                present.append(cls)
            else:
                per_class_iou[cls] = 0.0

        if len(present) > 0:
            miou = float(np.mean([per_class_iou[c] for c in present]))
        else:
            miou = 0.0

        val_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        val_acc = total_correct / total_points if total_points > 0 else 0.0

        # logging
        self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
        self.writer.add_scalar("val/epoch_acc", val_acc, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)

        print(f"验证轮次 {epoch}：损失 = {val_loss:.4f}, 准确率 = {val_acc:.4f}, mIoU = {miou:.4f}")
        print("[IoU per class]")
        for cls in range(self.config.NUM_CLASSES):
            if per_class_iou[cls] > 0 or cls in present:
                print(f"  [IoU] class {cls}: {per_class_iou[cls]:.4f}")

        if all_preds.size > 0:
            unique_preds, counts_preds = np.unique(all_preds, return_counts=True)
            print("Prediction distribution (val):", dict(zip(unique_preds.tolist(), counts_preds.tolist())))
        if all_labels.size > 0:
            unique_lbls, counts_lbls = np.unique(all_labels, return_counts=True)
            print("GT distribution (val):", dict(zip(unique_lbls.tolist(), counts_lbls.tolist())))

        # 评估结束后恢复训练权重
        if using_ema:
            self.ema.restore(self.model)

        return val_loss, val_acc, miou

    def train(self):
        print("开始训练...")
        try:
            for epoch in range(1, self.config.MAX_EPOCHS + 1):
                train_loss, train_acc = self.train_epoch(epoch)

                # scheduler
                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

                # validate periodically
                if epoch % self.config.EVAL_FREQ == 0:
                    val_loss, val_acc, val_miou = self.validate(epoch)

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
                        self.early_stop_counter = 0
                    else:
                        # 早停计数
                        if getattr(self.config, "EARLY_STOP_ENABLE", False):
                            self.early_stop_counter += 1
                            if self.early_stop_counter >= getattr(self.config, "EARLY_STOP_PATIENCE", 10):
                                print(f"[EarlyStopping] 验证 mIoU 连续 {self.early_stop_counter} 次未提升，提前停止训练。")
                                break

                # periodic save
                if epoch % self.config.SAVE_FREQ == 0:
                    save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"model_epoch_{epoch}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, save_path)
                    print(f"保存模型到 {save_path}")

        except Exception as e:
            error_save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"error_model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), error_save_path)
            print(f"训练中断，已保存当前模型到 {error_save_path}")
            raise e

        # final validate
        val_loss, val_acc, val_miou = self.validate(self.config.MAX_EPOCHS)
        print(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.writer.close()
