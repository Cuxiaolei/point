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
    LEARNING_RATE = 5e-5     # 稳定起步
    WEIGHT_DECAY  = 1e-5
    MAX_EPOCHS    = 100
    EVAL_FREQ     = 5
    SAVE_FREQ     = 10
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

    # ================== 数据处理 ==================
    LIMIT_POINTS  = False     # 为 False 时不要传 max_points
    MAX_POINTS    = 8000

    # 基础增强（由数据集/外层使用这些开关）
    ROTATION_AUG  = True
    SCALE_AUG     = True
    NOISE_AUG     = True
    TRANSFORM_AUG = True

    # 数据值规范化（数据集读取时使用）
    INPUT_NORM      = True      # 对每个样本的 XYZ 做零均值/单位范数
    COLOR_MODE      = "auto"    # {"auto","01","255"}: auto 自动判断是否需 /255
    NORMAL_UNIT     = True      # 法向量是否重新单位化
    INPUT_DIM       = 9         # [x,y,z | r,g,b | nx,ny,nz]，供模型/数据一致性参考

    # 训练期“反过拟合”策略（Trainer 会直接读取）
    POINT_DROPOUT_ENABLE = True
    POINT_DROPOUT_RATE   = 0.08

    POINTCUTMIX_ENABLE   = True
    POINTCUTMIX_PROB     = 0.12
    POINTCUTMIX_RATIO    = 0.20

    # ================== 稳定训练相关 ==================
    CLIP_NORM      = 2.0
    SEED           = 42
    DETERMINISTIC  = False

    # ================== 模型/损失增强 ==================
    ENABLE_DROPOUT       = True
    DROPOUT_RATE         = 0.3

    ENABLE_DROPPATH      = True
    DROPPATH_PROB        = 0.10

    ENABLE_FEATURE_DROP  = True
    FEATURE_DROP_PROB    = 0.15

    ENABLE_LABEL_SMOOTH  = True
    LABEL_SMOOTH_FACTOR  = 0.05

    ENABLE_FOCAL_LOSS    = True
    FOCAL_GAMMA          = 1.5
    FOCAL_ALPHA          = 0.25

    ENABLE_SOFT_DICE     = True  # 可选：soft dice（逐点分割友好）
    SOFT_DICE_EPS        = 1e-6
    SOFT_DICE_WEIGHT     = 0.5   # dice 的权重（和 CE/Focal 组合）

    ENABLE_TEMP_SCALING  = True  # 温度缩放（更平滑的 logits，利于稳定）
    TEMP_FACTOR          = 1.5

    # ====== 与模型文件（SGDAT / modules_sgdat）适配的关键开关 ======
    # —— 动态邻域 + 通道融合
    ENABLE_DYNAMIC_NEIGHBOR = True
    KNN_K_BASE              = 16
    KNN_K_MAX               = 32
    NEIGHBOR_ADAPTIVE       = True     # 动态邻域自适应策略开关
    ENABLE_CHANNEL_FUSION   = True     # 通道融合（Channel Fusion）

    # —— 语义引导的全尺度融合
    ENABLE_SEMANTIC_GUIDANCE = True
    SEMANTIC_AUX_LOSS_WEIGHT = 0.4     # 若模型内部有语义辅助头，可沿用该比重

    # —— 轻量化注意力：CAA + 线性 GVA
    ATTN_USE_CAA        = True   # CAA 中 CCC 子模块的通道相似度思想
    ATTN_USE_LINEAR_GVA = True   # 线性编码的 GVA，替代 CAE 中 softmax

    # ================== EMA / 早停 ==================
    EMA_ENABLE = True
    EMA_DECAY  = 0.999

    EARLY_STOP_ENABLE   = True
    EARLY_STOP_PATIENCE = 8

    # ================== 日志/保存 ==================
    LOG_FREQ       = 10
    NUM_WORKERS    = 4
    LOG_DIR        = f"logs/{_TS}"
    MODEL_SAVE_DIR = f"checkpoints/{_TS}"

    def __init__(self):
        try:
            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            if self.DEVICE.startswith("cuda"):
                torch.cuda.manual_seed_all(self.SEED)
            if self.DETERMINISTIC:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
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
