# trainer.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.cuda.amp  # 引入混合精度训练
import torch.cuda.amp as amp  # 使用 torch.amp 进行混合精度训练
from torch.amp import GradScaler, autocast

class Trainer:
    """模型训练器，优化后解决索引越界问题并提升内存效率"""

    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.DEVICE
        self.scaler = GradScaler() if self.device == 'cuda' else None  # 使用新的GradScaler API

        # 核心优化1：确保collate与数据集max_points同步，避免重复填充
        self.max_points = getattr(train_dataset, 'max_points', 20000)  # 从数据集获取最大点数
        print(f"训练器初始化：使用与数据集匹配的最大点数 {self.max_points}")

        # 自定义collate函数（优化版）
        def custom_collate(batch):
            """将批次中的点云统一到数据集定义的max_points长度"""
            points_list = [item[0] for item in batch]
            labels_list = [item[1] for item in batch]

            padded_points = []
            padded_labels = []
            for p, l in zip(points_list, labels_list):
                # 直接截断或填充到数据集定义的max_points（避免动态变化）
                if p.shape[0] > self.max_points:
                    # 截断到max_points（与数据集处理一致）
                    p = p[:self.max_points]
                    l = l[:self.max_points]
                elif p.shape[0] < self.max_points:
                    # 填充到max_points（用0填充点云，-1填充标签）
                    pad_size = self.max_points - p.shape[0]
                    p = torch.cat([p, torch.zeros(pad_size, p.shape[1], device=p.device)], dim=0)
                    l = torch.cat([l, torch.full((pad_size,), -1, device=l.device, dtype=l.dtype)], dim=0)

                padded_points.append(p)
                padded_labels.append(l)

            # 拼接成批次张量（确保形状固定）
            return torch.stack(padded_points), torch.stack(padded_labels)

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=True  # 丢弃最后一个不完整批次，避免形状异常
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
            pin_memory=True,
            collate_fn=custom_collate,
            drop_last=False
        )

        # 优化器和学习率调度器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.MAX_EPOCHS,
            eta_min=1e-5
        )

        # 核心优化2：启用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

        # 日志和模型保存
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.writer = SummaryWriter(config.LOG_DIR)

        # 记录最佳验证mIoU
        self.best_val_miou = 0.0

        # 打印配置
        config.print_config()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        # 初始化 tqdm 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.MAX_EPOCHS}")

        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=self.scaler is not None):  # 使用新的autocast API
                loss, logits = self.model.get_loss(points, labels, ignore_index=-1)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 计算准确率（排除填充的标签）
            mask = (labels != -1)
            if mask.sum() == 0:
                print(f"警告：第{epoch}轮第{batch_idx}批次无有效点，已跳过")
                continue

            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_points += mask.sum().item()
            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / mask.sum().item():.4f}",
                "有效点": f"{mask.sum().item()}"
            })

            # 写入TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            self.writer.add_scalar("train/batch_acc", correct / mask.sum().item(), global_step)

            # 核心优化4：及时清理内存
            del points, labels, loss, logits, preds, mask
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # 计算 epoch 级别的指标
        epoch_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        epoch_acc = total_correct / total_points if total_points > 0 else 0

        # 记录到TensorBoard
        self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        self.writer.add_scalar("train/epoch_acc", epoch_acc, epoch)

        print(f"训练轮次 {epoch}：损失 = {epoch_loss:.4f}, 准确率 = {epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """验证模型性能（优化索引安全和效率）"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        # 混淆矩阵（使用int64避免溢出）
        conf_matrix = np.zeros(
            (self.config.NUM_CLASSES, self.config.NUM_CLASSES),
            dtype=np.int64
        )

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"验证轮次 {epoch}")

            for batch_idx, (points, labels) in enumerate(pbar):
                # 移动数据到设备
                points = points.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 检查数据有效性
                if points.shape[0] == 0:
                    print(f"警告：验证第{batch_idx}批次为空，已跳过")
                    continue

                # 前向传播（混合精度）
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    loss, logits = self.model.get_loss(points, labels, ignore_index=-1)
                preds = logits.argmax(dim=-1)

                # 计算指标（排除填充的标签）
                mask = (labels != -1)
                if mask.sum() == 0:
                    print(f"警告：验证第{batch_idx}批次无有效点，已跳过")
                    continue

                correct = (preds[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_points += mask.sum().item()
                total_loss += loss.item()

                # 核心优化5：安全更新混淆矩阵（限制标签范围）
                batch_labels = labels[mask].cpu().numpy().flatten()
                batch_preds = preds[mask].cpu().numpy().flatten()

                # 确保标签在有效范围内（避免索引越界）
                valid_cls_mask = (batch_labels < self.config.NUM_CLASSES) & (batch_labels >= 0)
                batch_labels = batch_labels[valid_cls_mask]
                batch_preds = batch_preds[valid_cls_mask]

                if len(batch_labels) > 0:
                    batch_conf = confusion_matrix(
                        batch_labels,
                        batch_preds,
                        labels=np.arange(self.config.NUM_CLASSES)
                    )
                    conf_matrix += batch_conf

                # 打印批次信息
                batch_acc = correct / mask.sum().item()
                pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_acc": f"{batch_acc:.4f}",
                    "有效点": f"{mask.sum().item()}"
                })

                # 清理内存
                del points, labels, loss, logits, preds, mask
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        # 计算mIoU（处理可能的零分母）
        ious = []
        for cls in range(self.config.NUM_CLASSES):
            tp = conf_matrix[cls, cls]
            fp = conf_matrix[:, cls].sum() - tp
            fn = conf_matrix[cls, :].sum() - tp

            # 避免除以零（当该类别无样本时）
            if tp + fp + fn == 0:
                iou = 0.0
            else:
                iou = tp / (tp + fp + fn)
            ious.append(iou)
            self.writer.add_scalar(f"val/iou_{cls}", iou, epoch)

        miou = np.mean(ious) if ious else 0.0

        # 计算整体指标
        val_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        val_acc = total_correct / total_points if total_points > 0 else 0

        # 记录到TensorBoard
        self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
        self.writer.add_scalar("val/epoch_acc", val_acc, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)

        print(f"验证轮次 {epoch}：损失 = {val_loss:.4f}, 准确率 = {val_acc:.4f}, mIoU = {miou:.4f}")
        return val_loss, val_acc, miou

    def train(self):
        """完整训练过程（增加鲁棒性处理）"""
        print("开始训练...")

        try:  # 捕获异常，确保训练中断时能保存模型
            for epoch in range(1, self.config.MAX_EPOCHS + 1):
                # 训练一个轮次
                train_loss, train_acc = self.train_epoch(epoch)

                # 学习率调度
                self.scheduler.step()
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

                # 定期验证
                if epoch % self.config.EVAL_FREQ == 0:
                    val_loss, val_acc, val_miou = self.validate(epoch)

                    # 保存最佳模型
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

                # 定期保存模型
                if epoch % self.config.SAVE_FREQ == 0:
                    save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"model_epoch_{epoch}.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, save_path)
                    print(f"保存模型到 {save_path}")

        except Exception as e:
            # 异常时保存当前模型
            error_save_path = os.path.join(self.config.MODEL_SAVE_DIR, f"error_model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), error_save_path)
            print(f"训练中断，已保存当前模型到 {error_save_path}")
            raise e  # 重新抛出异常，显示错误信息

        # 训练结束，最后验证一次
        val_loss, val_acc, val_miou = self.validate(self.config.MAX_EPOCHS)

        print(f"训练完成！最佳验证mIoU = {self.best_val_miou:.4f}")
        self.writer.close()
