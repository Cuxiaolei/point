import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
import torch.nn.utils as nn_utils
from tabulate import tabulate  # 新增

class Trainer:
    """
    训练器（增强版）
    - 自动计算 class weights 并传入 model.get_loss
    - 验证阶段收集 preds/labels，打印分布与 per-class IoU（只对存在类求均值）
    - 数值检测（NaN/Inf）并保存异常模型快照
    - AMP + grad clipping（可通过 config 控制）
    """

    def __init__(self, model, config, train_dataset, val_dataset, class_weights=None):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.DEVICE

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_miou": [],
            "val_loss": [],
            "val_acc": [],
            "val_miou": []
        }

        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights = class_weights.detach().clone().to(self.device, dtype=torch.float32)
            else:
                self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            self.class_weights = None

        self.scaler = GradScaler() if self.device.startswith('cuda') else None

        # 最大点数（只在 LIMIT_POINTS=True 时生效）
        self.max_points = getattr(config, 'MAX_POINTS', 20000)
        if config.LIMIT_POINTS:
            print(f"训练器初始化：限制最大点数 {self.max_points}")

        # 自定义 collate
        def custom_collate(batch):
            points_list = [item[0] for item in batch]
            labels_list = [item[1] for item in batch]
            if not config.LIMIT_POINTS:
                return torch.stack(points_list), torch.stack(labels_list)

            padded_points, padded_labels = [], []
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

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=False
        )

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
        self.clip_norm = getattr(config, 'CLIP_NORM', 2.0)

        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.writer = SummaryWriter(config.LOG_DIR)
        self.best_val_miou = 0.0
        self.class_weights = self._compute_class_weights()
        print(f"[Trainer] class_weights: {self.class_weights}")
        config.print_config()

    def _compute_class_weights(self):
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
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.MAX_EPOCHS}")
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=self.scaler is not None):
                loss, logits, stats = self.model.get_loss(
                    points, labels,
                    class_weights=self.class_weights if hasattr(self, 'class_weights') else None,
                    ignore_index=-1,
                    aux_weight=0.4,
                    label_smoothing=0.05,
                    focal_gamma=1.5,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"bad_loss_epoch{epoch}_batch{batch_idx}.pth")
                torch.save(self.model.state_dict(), save_path)
                raise RuntimeError(f"Encountered bad loss (NaN/Inf) at epoch {epoch} batch {batch_idx}. Model saved to {save_path}")

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

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

            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            self.writer.add_scalar("train/batch_acc", correct / mask.sum().item(), global_step)

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

        miou = float(np.mean([per_class_iou[c] for c in present])) if len(present) > 0 else 0.0
        val_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        val_acc = total_correct / total_points if total_points > 0 else 0.0

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

        return val_loss, val_acc, miou

    def train(self):
        print("开始训练...")
        try:
            for epoch in range(1, self.config.MAX_EPOCHS + 1):
                train_loss, train_acc = self.train_epoch(epoch)

                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

                val_loss, val_acc, val_miou = float('nan'), float('nan'), float('nan')
                if epoch % self.config.EVAL_FREQ == 0:
                    val_loss, val_acc, val_miou = self.validate(epoch)
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

                if epoch % self.config.SAVE_FREQ == 0:
                    save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"model_epoch_{epoch}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, save_path)
                    print(f"保存模型到 {save_path}")

                # 记录历史
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["train_miou"].append(float('nan'))  # 如果有训练mIoU可替换
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_miou"].append(val_miou)

        except Exception as e:
            error_save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"error_model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), error_save_path)
            print(f"训练中断，已保存当前模型到 {error_save_path}")
            raise e

        val_loss, val_acc, val_miou = self.validate(self.config.MAX_EPOCHS)
        print(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")

        # 打印与保存总结
        self.print_summary()

        self.writer.close()

    def print_summary(self):
        """打印并保存训练/验证指标表格"""
        headers = ["Epoch", "Train Loss", "Train Acc", "Train mIoU", "Val Loss", "Val Acc", "Val mIoU"]
        table_data = []
        for i in range(len(self.history["epoch"])):
            table_data.append([
                self.history["epoch"][i],
                f"{self.history['train_loss'][i]:.4f}" if not np.isnan(self.history['train_loss'][i]) else "-",
                f"{self.history['train_acc'][i]:.4f}" if not np.isnan(self.history['train_acc'][i]) else "-",
                f"{self.history['train_miou'][i]:.4f}" if not np.isnan(self.history['train_miou'][i]) else "-",
                f"{self.history['val_loss'][i]:.4f}" if not np.isnan(self.history['val_loss'][i]) else "-",
                f"{self.history['val_acc'][i]:.4f}" if not np.isnan(self.history['val_acc'][i]) else "-",
                f"{self.history['val_miou'][i]:.4f}" if not np.isnan(self.history['val_miou'][i]) else "-"
            ])
        summary_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print("\n训练结果总结：\n")
        print(summary_str)
        log_path = os.path.join(self.config.LOG_DIR, "summary.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(summary_str)
        print(f"\n训练总结已保存到: {log_path}")
