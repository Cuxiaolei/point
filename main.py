import torch
import numpy as np
import os
import collections

from models.sgdat import SGDAT
from datasets.point_cloud_dataset import PointCloudDataset
from trainer import Trainer
from config import Config


def infer_num_classes_from_dataset(dataset):
    """从数据集推断类别数"""
    all_labels = set()
    for scene in dataset.scene_list:
        seg_path = os.path.join(scene, 'segment20.npy')
        if os.path.exists(seg_path):
            arr = np.load(seg_path)
            arr = arr[arr >= 0]
            all_labels.update(arr.tolist())
    return max(all_labels) + 1 if all_labels else 1


def compute_class_weights(dataset, num_classes):
    """根据数据集计算类别权重"""
    cnt = collections.Counter()
    for scene in dataset.scene_list:
        seg_path = os.path.join(scene, 'segment20.npy')
        if os.path.exists(seg_path):
            arr = np.load(seg_path)
            arr = arr[arr != -1]
            cnt.update(arr.tolist())

    counts = np.array([cnt.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    print(f"[Trainer] class counts: {counts}")
    print(f"[Trainer] class_weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def main():
    torch.manual_seed(42)
    config = Config()

    # 构建数据集（参数与 PointCloudDataset 定义一致）
    train_dataset = PointCloudDataset(
        root_dir=os.path.join(config.DATA_ROOT, "train"),
        num_classes=config.NUM_CLASSES,
        transform_aug=config.TRANSFORM_AUG,
        rotation_aug=config.ROTATION_AUG,
        scale_aug=config.SCALE_AUG,
        noise_aug=config.NOISE_AUG,
        limit_points=config.LIMIT_POINTS,
        max_points=config.MAX_POINTS
    )

    val_dataset = PointCloudDataset(
        root_dir=os.path.join(config.DATA_ROOT, "val"),
        num_classes=config.NUM_CLASSES,
        transform_aug=False,
        rotation_aug=False,
        scale_aug=False,
        noise_aug=False,
        limit_points=config.LIMIT_POINTS,
        max_points=config.MAX_POINTS
    )

    # 更新类别数
    num_classes = infer_num_classes_from_dataset(train_dataset)
    config.NUM_CLASSES = num_classes
    print(f"[INFO] Inferred num_classes = {num_classes}")

    # 类别权重
    class_weights = compute_class_weights(train_dataset, num_classes).to(config.DEVICE)

    # 初始化模型
    model = SGDAT(num_classes=config.NUM_CLASSES, base_dim=64, max_points=config.MAX_POINTS, debug=True)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 初始化训练器
    trainer = Trainer(model, config, train_dataset, val_dataset, class_weights=class_weights)
    trainer.train()


if __name__ == "__main__":
    main()
