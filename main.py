import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import Config
from point_cloud_dataset import PointCloudDataset
from sgdat import SGDAT
from trainer import Trainer


def seed_all(seed=42):
    """固定随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_weights(module):
    """初始化模型权重，防止数值不稳定"""
    if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def main():
    # 1. 固定随机种子
    seed_all(42)

    # 2. 加载配置
    config = Config()

    # 3. 检查数据路径
    if not os.path.exists(config.DATA_ROOT):
        print(f"[ERROR] 数据集路径不存在: {config.DATA_ROOT}")
        sys.exit(1)

    # 4. 数据集
    train_dataset = PointCloudDataset(
        data_root=config.DATA_ROOT,
        split="train",
        max_points=config.MAX_POINTS if config.LIMIT_POINTS else None
    )
    val_dataset = PointCloudDataset(
        data_root=config.DATA_ROOT,
        split="val",
        max_points=config.MAX_POINTS if config.LIMIT_POINTS else None
    )

    # 5. 自动推断类别数
    if config.NUM_CLASSES is None:
        all_labels = []
        for ds in [train_dataset, val_dataset]:
            for scene in ds.scene_list:
                seg_path = os.path.join(scene, "segment20.npy")
                if os.path.exists(seg_path):
                    arr = np.load(seg_path)
                    arr = arr[arr >= 0]
                    all_labels.extend(arr.tolist())
        if all_labels:
            config.NUM_CLASSES = max(all_labels) + 1
        else:
            config.NUM_CLASSES = 1
    print(f"[INFO] Inferred num_classes = {config.NUM_CLASSES}")

    # 6. 初始化模型
    model = SGDAT(num_classes=config.NUM_CLASSES)
    model.apply(init_weights)  # 权重初始化

    # 7. 打印参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 8. 类别权重（防 NaN/Inf）
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    for scene in train_dataset.scene_list:
        seg_path = os.path.join(scene, "segment20.npy")
        if os.path.exists(seg_path):
            arr = np.load(seg_path)
            arr = arr[arr != -1]
            binc = np.bincount(arr, minlength=config.NUM_CLASSES)
            counts += binc
    inv = 1.0 / (counts + 1e-6)
    weights = inv / inv.sum() * float(config.NUM_CLASSES)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)
    print(f"[Trainer] class counts: {counts}")
    print(f"[Trainer] class_weights: {class_weights}")

    # 9. 初始化 Trainer（Trainer 内部已做梯度裁剪 & scheduler 顺序修复）
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights
    )

    # 10. 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
