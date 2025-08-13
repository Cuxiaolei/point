import torch
import numpy as np
import os
import collections

from models.sgdat import SGDAT
from datasets.point_cloud_dataset import PointCloudDataset, PointCloudTransform
from trainer import Trainer
from config import Config


def infer_num_classes_from_dataset(train_dataset, val_dataset):
    """从训练集和验证集联合推断类别数"""
    all_labels = set()
    for ds in [train_dataset, val_dataset]:
        for scene in ds.scene_list:
            seg_path = os.path.join(scene, 'segment20.npy')
            if os.path.exists(seg_path):
                arr = np.load(seg_path)
                arr = arr[arr >= 0]  # 忽略 pad
                all_labels.update(arr.tolist())
    if not all_labels:
        return 1
    return max(all_labels) + 1


def compute_class_weights(train_dataset, num_classes):
    """根据训练集计算类别权重"""
    cnt = collections.Counter()
    for scene in train_dataset.scene_list:
        seg_path = os.path.join(scene, 'segment20.npy')
        if os.path.exists(seg_path):
            arr = np.load(seg_path)
            arr = arr[arr != -1]
            cnt.update(arr.tolist())

    counts = np.array([cnt.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    weights = counts.sum() / (counts + 1e-6)  # 防止除 0
    weights = weights / weights.mean()  # 归一化
    print(f"[Trainer] class counts: {counts}")
    print(f"[Trainer] class_weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def main():
    torch.manual_seed(42)  # 保证可复现

    # 加载配置
    config = Config()

    # 数据增强
    transform = PointCloudTransform(
        rotation=config.ROTATION_AUG,
        scale=config.SCALE_AUG,
        noise=config.NOISE_AUG,
        translate=config.TRANSFORM_AUG
    )

    # 数据集
    train_dataset = PointCloudDataset(config.DATA_ROOT, split="train", transform=transform, max_points=config.MAX_POINTS)
    val_dataset = PointCloudDataset(config.DATA_ROOT, split="val", transform=None, max_points=config.MAX_POINTS)

    # 类别数
    num_classes = infer_num_classes_from_dataset(train_dataset, val_dataset)
    print(f"[INFO] Inferred num_classes = {num_classes}")
    config.NUM_CLASSES = num_classes

    # 类别权重
    class_weights = compute_class_weights(train_dataset, num_classes).to(config.DEVICE)

    # 初始化模型
    model = SGDAT(num_classes=config.NUM_CLASSES, base_dim=64, max_points=config.MAX_POINTS, debug=True)

    # 打印参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 初始化 Trainer
    trainer = Trainer(model, config, train_dataset, val_dataset, class_weights=class_weights)
    trainer.train()


if __name__ == "__main__":
    main()
