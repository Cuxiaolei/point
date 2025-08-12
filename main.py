import torch

from models import DATSeg
from models.sgdat import SGDATSeg

from datasets.point_cloud_dataset import PointCloudDataset, PointCloudTransform
from trainer import Trainer
from config import Config


def main():
    # 加载配置
    config = Config(
        NUM_CLASSES=13,  # 根据实际数据集调整
        BATCH_SIZE=1,
        MAX_EPOCHS=100,
        LEARNING_RATE=1e-3,
        MAX_POINTS=8000  # 修改为合适的最大点数
    )

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

    # 初始化DAT模型
    # model = DATSeg(num_classes=config.NUM_CLASSES)

    # 初始化SGDAT模型
    model = SGDATSeg(num_classes=config.NUM_CLASSES)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 初始化训练器并开始训练
    trainer = Trainer(model, config, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    main()
