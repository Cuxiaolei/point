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
    config.print_config()

    if not os.path.exists(config.DATA_ROOT):
        print(f"[ERROR] 数据集路径不存在: {config.DATA_ROOT}")
        sys.exit(1)

    # 数据集构造（与 point_cloud_dataset.py 的新增强对齐）
    common_ds_kwargs = dict(
        rotation=config.ROTATION_AUG,
        scale=config.SCALE_AUG,
        noise=config.NOISE_AUG,
        translate=config.TRANSLATE_AUG,
        limit_points=config.LIMIT_POINTS,
        max_points=config.MAX_POINTS,
        normalize=config.INPUT_NORM,
        color_mode=config.COLOR_MODE,
        normal_unit=config.NORMAL_UNIT,
        num_classes=config.NUM_CLASSES,
        point_dropout=config.POINT_DROPOUT_ENABLE,
        point_dropout_rate=config.POINT_DROPOUT_RATE,
        cutmix=config.POINTCUTMIX_ENABLE,
        cutmix_prob=config.POINTCUTMIX_PROB,
        cutmix_ratio=config.POINTCUTMIX_RATIO,
        color_jitter=getattr(config, "COLOR_JITTER_ENABLE", False),
        color_jitter_params=getattr(config, "COLOR_JITTER_PARAMS", None),
        normal_noise_std=getattr(config, "NORMAL_NOISE_STD", 0.0),
    )
    train_dataset = PointCloudDataset(data_root=config.DATA_ROOT, split="train", **common_ds_kwargs)
    val_dataset = PointCloudDataset(data_root=config.DATA_ROOT, split="val", **common_ds_kwargs)

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

    # 初始化模型（cfg=config 以启用动态邻域、语义引导等新功能）
    model = SGDAT(
        in_dim=config.IN_DIM,
        base_dim=config.base_dim,
        num_classes=config.NUM_CLASSES,
        bn_eps=config.bn_eps,
        bn_momentum=config.bn_momentum,
        dropout_p=config.dropout_p,
        use_channel_ccc=config.use_channel_ccc,
        use_linear_gva=config.use_linear_gva,
        use_dynamic_fusion=config.use_dynamic_fusion,
        use_semantic_guided_fusion=config.use_semantic_guided_fusion,
        gva_lin_embed=config.GVA_LIN_EMBED,
        dyn_neighbors=config.dyn_neighbors,
        dyn_tau=config.dyn_tau,
        logit_temp=config.logit_temp,
        debug=config.DEBUG
    )
    model.apply(init_weights)

    # 打印参数数量
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

    # 创建 Trainer
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
