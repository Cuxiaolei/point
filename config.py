import os
from datetime import datetime
import torch
# 111111


class Config:
    # ================== 数据路径 ==================
    DATA_ROOT = "/root/my/point/data/tower"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")

    # ================== 模型参数 ==================
    NUM_CLASSES = 3
    BATCH_SIZE = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 100
    EVAL_FREQ = 5
    SAVE_FREQ = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 限制每个样本点数
    LIMIT_POINTS = False
    MAX_POINTS = 8000

    # 数据增强
    ROTATION_AUG = True
    SCALE_AUG = True
    NOISE_AUG = True
    TRANSFORM_AUG = True

    # 日志与保存路径
    LOG_FREQ = 10
    CLIP_NORM = 2.0
    NUM_WORKERS = 4
    LOG_DIR = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_SAVE_DIR = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ================== 反过拟合策略开关 ==================
    ENABLE_DROPOUT = True  # 控制Dropout
    DROPOUT_RATE = 0.3

    ENABLE_DROPPATH = True  # 控制DropPath
    DROPPATH_PROB = 0.1

    ENABLE_FEATURE_DROP = True  # 随机丢弃特征
    FEATURE_DROP_PROB = 0.15

    ENABLE_LABEL_SMOOTH = True  # 标签平滑
    LABEL_SMOOTH_FACTOR = 0.1

    ENABLE_FOCAL_LOSS = True  # Focal Loss
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25

    ENABLE_TEMP_SCALING = True  # 温度缩放
    TEMP_FACTOR = 1.5


    def print_config(self):
        print("==================================================")
        print("训练配置:")
        for k, v in vars(self).items():
            if k.startswith("_"):  # 跳过私有
                continue
            if callable(v):
                continue
            print(f"{k}: {v}")
        print("==================================================")
