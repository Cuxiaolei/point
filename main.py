import torch
import numpy as np
import os
import collections

from models.sgdat import SGDAT
from datasets.point_cloud_dataset import PointCloudDataset
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
    weights = weights / (weights.mean() + 1e-12)  # 归一化
    print(f"[Trainer] class counts: {counts}")
    print(f"[Trainer] class_weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def main():
    torch.manual_seed(42)  # 保证可复现

    # 加载配置
    config = Config()

    # 数据集（保持你的 PointCloudDataset 的参数签名）
    train_dataset = PointCloudDataset(
        config.DATA_ROOT,
        split="train",
        rotation=config.ROTATION_AUG,
        scale=config.SCALE_AUG,
        noise=config.NOISE_AUG,
        translate=config.TRANSFORM_AUG,
        max_points=config.MAX_POINTS if config.LIMIT_POINTS else None
    )
    val_dataset = PointCloudDataset(
        config.DATA_ROOT,
        split="val",
        rotation=False,
        scale=False,
        noise=False,
        translate=False,
        max_points=config.MAX_POINTS if config.LIMIT_POINTS else None
    )

    # 类别数自动推断（仍然会被 config.NUM_CLASSES 覆盖/更新）
    num_classes = infer_num_classes_from_dataset(train_dataset, val_dataset)
    print(f"[INFO] Inferred num_classes = {num_classes}")
    config.NUM_CLASSES = num_classes

    # 类别权重
    class_weights = compute_class_weights(train_dataset, num_classes).to(config.DEVICE)

    # =========================
    # 初始化模型：所有“反过拟合”参数都从 Config 读取，并显式传入 SGDAT
    # =========================
    dropout_p = config.DROPOUT_RATE if getattr(config, "ENABLE_DROPOUT", False) else 0.0
    droppath_prob = config.DROPPATH_PROB if getattr(config, "ENABLE_DROPPATH", False) else 0.0
    # 若只提供了一个 FEATURE_DROP_PROB，就对 RGB/Normal 使用同一个概率
    feat_drop_p = config.FEATURE_DROP_PROB if getattr(config, "ENABLE_FEATURE_DROP", False) else 0.0
    logit_temp = config.TEMP_FACTOR if getattr(config, "ENABLE_TEMP_SCALING", False) else 1.0
    use_channel_ccc = getattr(config, "ENABLE_CHANNEL_CCC", False)

    model = SGDAT(
        num_classes=config.NUM_CLASSES,
        base_dim=64,
        max_points=config.MAX_POINTS if config.LIMIT_POINTS else None,
        debug=True,
        dropout_p=dropout_p,
        droppath_prob=droppath_prob,
        drop_rgb_p=feat_drop_p,
        drop_normal_p=feat_drop_p,
        logit_temp=logit_temp,
        use_channel_ccc=use_channel_ccc
    )

    # 打印参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 初始化 Trainer（反过拟合策略：如标签平滑与 Focal 在 Trainer/损失处启用）
    trainer = Trainer(model, config, train_dataset, val_dataset, class_weights=class_weights)
    trainer.train()


if __name__ == "__main__":
    main()
