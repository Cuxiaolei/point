import torch

from models import DATSeg
from models.sgdat import SGDATSeg

from datasets.point_cloud_dataset import PointCloudDataset, PointCloudTransform
from trainer import Trainer
from config import Config

# main.py - 在创建 dataset 后，设置 config.NUM_CLASSES
import numpy as np
import os
import numpy as np, os, collections

def infer_num_classes(data_root):
    labels_set = set()
    for split in ['train', 'val']:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for scene in os.listdir(split_dir):
            seg_path = os.path.join(split_dir, scene, 'segment20.npy')
            if os.path.exists(seg_path):
                arr = np.load(seg_path)
                labels_set.update(np.unique(arr).tolist())
    # 忽略 pad label -1
    labels = [int(x) for x in labels_set if int(x) >= 0]
    if not labels:
        return 1
    return max(labels) + 1


def main():


    # 加载配置
    config = Config()

    # 初始化数据增强
    transform = PointCloudTransform(
        rotation=config.ROTATION_AUG,
        scale=config.SCALE_AUG,
        noise=config.NOISE_AUG,
        translate=config.TRANSFORM_AUG
    )

    # 加载数据集
    train_dataset = PointCloudDataset(
        root_dir=config.DATA_ROOT,
        split="train",
        transform=transform,
        max_points=config.MAX_POINTS  # 使用配置中的点数
    )

    val_dataset = PointCloudDataset(
        root_dir=config.DATA_ROOT,
        split="val",
        transform=None,  # 验证集不使用数据增强
        max_points=config.MAX_POINTS  # 使用配置中的点数
    )

    num_classes = infer_num_classes(config.DATA_ROOT)
    print(f"[INFO] Inferred num_classes = {num_classes}")
    config.NUM_CLASSES = num_classes

    # 初始化DAT模型
    # model = DATSeg(num_classes=config.NUM_CLASSES)

    # 初始化SGDAT模型
    model = SGDATSeg(num_classes=config.NUM_CLASSES, input_dim=9, base_dim=64, max_points=config.MAX_POINTS, debug=True)


    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    cnt = collections.Counter()
    for scene in train_dataset.scene_list:
        arr = np.load(os.path.join(scene, 'segment20.npy'))
        vals = arr.flatten()
        vals = vals[vals != -1]
        cnt.update(map(int, vals.tolist()))
    print("[Label distribution] (label: count)")
    for k, v in sorted(cnt.items()):
        print(k, v)

    # 初始化训练器并开始训练
    trainer = Trainer(model, config, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    main()
