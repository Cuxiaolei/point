import os
from datetime import datetime
import torch
import random
import numpy as np

# 统一时间戳，确保 LOG_DIR 与 MODEL_SAVE_DIR 一致
_TS = datetime.now().strftime('%Y%m%d_%H%M%S')


class Config:
    # ================== 路径 ==================
    DATA_ROOT = "/root/my/point/data/tower"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR   = os.path.join(DATA_ROOT, "val")

    # ================== 训练超参 ==================
    NUM_CLASSES   = 3
    BATCH_SIZE    = 5
    LEARNING_RATE = 5e-5     # 稳定起步，之前 1e-4 容易发散；Trainer 用这个字段
    BASE_LR       = 5e-5     # 仅作记录，实际优化器读取 LEARNING_RATE
    WEIGHT_DECAY  = 1e-5
    MAX_EPOCHS    = 100
    EVAL_FREQ     = 5         # 与 Trainer 对齐：每 5 轮验证一次
    SAVE_FREQ     = 10
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

    # ================== 数据处理 ==================
    # 固定 / 截断每样本点数；Trainer 会读取 LIMIT_POINTS / MAX_POINTS
    LIMIT_POINTS  = False
    MAX_POINTS    = 8000

    # 基础增强（由数据集/外层使用这些开关）
    ROTATION_AUG  = True
    SCALE_AUG     = True
    NOISE_AUG     = True
    TRANSFORM_AUG = True

    # 训练期“反过拟合”策略（Trainer 会直接读取）
    POINT_DROPOUT_ENABLE = True   # 随机将部分有效点置为无效（label=-1）
    POINT_DROPOUT_RATE   = 0.08   # 建议 0.05~0.15

    POINTCUTMIX_ENABLE   = True   # 批内点云 CutMix
    POINTCUTMIX_PROB     = 0.12
    POINTCUTMIX_RATIO    = 0.20

    # ================== 稳定训练相关 ==================
    INPUT_NORM     = True      # 对每个样本的 XYZ 做零均值/单位方差标准化（由数据侧使用）
    CLIP_NORM      = 2.0       # Trainer 中会用 nn_utils.clip_grad_norm_
    CLIP_GRAD_NORM = 2.0       # 仅作记录；实际由 CLIP_NORM 生效
    SEED           = 42        # 便于复现
    DETERMINISTIC  = False     # 如需完全确定性可改 True（会稍慢）

    # ================== 轻量注意力 / 融合组件（模型侧读取） ==================
    ENABLE_CHANNEL_CCC   = True   # CAA 的 CCC 子模块-通道相似度
    ENABLE_DYNAMIC_FUSION = True  # 动态邻域-通道融合
    ENABLE_LINEAR_GVA    = True   # 线性 GVA（替代 Softmax 归一化）

    # 动态邻域参数
    DYN_NEIGHBORS  = 16
    DYN_MIN_RADIUS = 0.02
    DYN_MAX_RADIUS = 0.30

    # ================== 正则/损失增强（模型 get_loss 可使用） ==================
    ENABLE_DROPOUT       = True
    DROPOUT_RATE         = 0.3

    ENABLE_DROPPATH      = True
    DROPPATH_PROB        = 0.10

    ENABLE_FEATURE_DROP  = True
    FEATURE_DROP_PROB    = 0.15

    ENABLE_LABEL_SMOOTH  = True
    LABEL_SMOOTH_FACTOR  = 0.05   # Trainer 默认传 0.05，这里保持一致

    ENABLE_FOCAL_LOSS    = True
    FOCAL_GAMMA          = 1.5    # Trainer 默认传 1.5
    FOCAL_ALPHA          = 0.25

    ENABLE_TEMP_SCALING  = True
    TEMP_FACTOR          = 1.5

    # ================== EMA / 早停 ==================
    EMA_ENABLE = True        # Trainer 会读取并创建 ModelEMA
    EMA_DECAY  = 0.999

    EARLY_STOP_ENABLE   = True
    EARLY_STOP_PATIENCE = 8  # 连续 8 次验证 mIoU 不提升则早停

    # ================== 日志/保存 ==================
    LOG_FREQ       = 10
    NUM_WORKERS    = 4
    LOG_DIR        = f"logs/{_TS}"
    MODEL_SAVE_DIR = f"checkpoints/{_TS}"

    def __init__(self):
        """可选：在实例化时设置随机种子，便于复现"""
        try:
            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            if self.DEVICE.startswith("cuda"):
                torch.cuda.manual_seed_all(self.SEED)
            # 完全确定性（会影响性能），按需启用
            if self.DETERMINISTIC:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            # 环境未就绪时静默跳过
            pass

    def print_config(self):
        print("==================================================")
        print("训练配置:")
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            print(f"{k}: {v}")
        print("==================================================")
