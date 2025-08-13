import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import Config
from datasets.point_cloud_dataset import PointCloudDataset
from models.sgdat import SGDAT
from trainer import Trainer


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_weights(module):
    if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def main():
    seed_all(42)
    config = Config()

    if not os.path.exists(config.DATA_ROOT):
        print(f"[ERROR] 数据集路径不存在: {config.DATA_ROOT}")
        sys.exit(1)

    # 数据集 —— 当 LIMIT_POINTS=False 时，不传 max_points
    train_kwargs = dict(data_root=config.DATA_ROOT, split="train")
    val_kwargs   = dict(data_root=config.DATA_ROOT, split="val")
    if config.LIMIT_POINTS:
        train_kwargs["max_points"] = config.MAX_POINTS
        val_kwargs["max_points"] = config.MAX_POINTS

    train_dataset = PointCloudDataset(**train_kwargs)
    val_dataset   = PointCloudDataset(**val_kwargs)

    # 自动推断类别数
    if config.NUM_CLASSES is None:
        all_labels = []
        for ds in [train_dataset, val_dataset]:
            for scene in ds.scene_list:
                seg_path = os.path.join(scene, "segment20.npy")
                if os.path.exists(seg_path):
                    arr = np.load(seg_path)
                    arr = arr[arr >= 0]
                    all_labels.extend(arr.tolist())
        config.NUM_CLASSES = max(all_labels) + 1 if all_labels else 1
    print(f"[INFO] Inferred num_classes = {config.NUM_CLASSES}")

    # 初始化模型
    model = SGDAT(num_classes=config.NUM_CLASSES, cfg=config)
    model.apply(init_weights)

    # 打印参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 类别权重
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

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights
    )

    trainer.train()


if __name__ == "__main__":
    main()
