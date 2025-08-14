# config.py
import os
import time
import random
import numpy as np
import torch

_TS = time.strftime("%Y%m%d_%H%M%S")


class Config:
    """
    训练/数据/模型的集中配置。
    与 trainer.py / point_cloud_dataset.py / sgdat.py 完整对齐。
    """

    # ================== 设备与运行 ==================
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True              # trainer 会根据是否有 AMP 自动启用
    DETECT_ANOMALY = False              # 仅在排查异常时打开（会变慢）
    NUM_WORKERS = 4

    # 当单步 loss 非法或过大时，跳过该步（并保存快照）
    SKIP_STEP_LOSS_THRESH = 1e6

    # ================== 数据路径与数据集 ==================
    # 根目录下期望存在 train/val 分割索引或你自己的扫描列表逻辑
    DATA_ROOT = "/root/my/point/data/tower"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TRAIN_LIST = None                   # 如使用 txt 索引，可在 DataSet 内读取
    VAL_LIST = None

    # 数据形态与归一化
    NUM_CLASSES = 3
    COLOR_MODE = "rgb"                  # {"none","rgb"}
    NORMAL_UNIT = True                  # 将法向量单位化
    INPUT_NORM = True                   # 对坐标/颜色做归一化（DataSet 内部实现）

    # 采样与点数控制（与 DataLoader 的 collate 对齐）
    LIMIT_POINTS = -1                 # 每个样本训练时最多取多少点（-1 表示不裁）
    MAX_POINTS = 12000                  # 模型内部用于位置嵌入/插值等的上限（与 sgdat 对齐）  # noqa

    # 数据增强（与 point_cloud_dataset.py 对齐）
    ROTATION_AUG = True
    SCALE_AUG = True
    NOISE_AUG = True
    TRANSLATE_AUG = True

    # 噪声/缩放等幅度（DataSet 内已有默认值，可按需覆盖）
    ROTATION_AXIS = "z"                 # {"x","y","z","all"} 若数据竖直方向明确可用 "z"
    SCALE_MIN, SCALE_MAX = 0.9, 1.1
    NOISE_STD = 0.005
    TRANSLATE_RANGE = 0.02

    # ================== 训练超参 ==================
    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_NORM = 2.0                      # 梯度裁剪阈值
    EVAL_FREQ = 1                        # 每多少轮做一次验证
    SAVE_FREQ = 1                        # 每多少轮保存一次

    # ================== 损失增强/稳定项（trainer 读取） ==================
    ENABLE_LABEL_SMOOTH = True
    LABEL_SMOOTH_FACTOR = 0.05

    ENABLE_FOCAL_LOSS = True
    FOCAL_GAMMA = 1.5
    FOCAL_ALPHA = 0.25

    ENABLE_SOFT_DICE = True
    SOFT_DICE_EPS = 1e-6
    SOFT_DICE_WEIGHT = 0.5

    ENABLE_TEMP_SCALING = True
    TEMP_FACTOR = 1.5

    # 语义辅助头（若模型内部用到）
    SEMANTIC_AUX_LOSS_WEIGHT = 0.4

    # ================== 训练期“反过拟合”策略（trainer 里实现） ==================
    POINT_DROPOUT_ENABLE = True
    POINT_DROPOUT_RATE = 0.08

    POINTCUTMIX_ENABLE = True
    POINTCUTMIX_PROB = 0.12
    POINTCUTMIX_RATIO = 0.20

    # ================== EMA / Early Stop ==================
    EMA_ENABLE = True
    EMA_DECAY = 0.999

    EARLY_STOP_ENABLE = True
    EARLY_STOP_PATIENCE = 8

    # ================== 模型（SGDAT）相关开关 ==================
    # 对齐 sgdat.SGDAT 的 __init__ 参数命名（通过 cfg 传入）：
    #   use_channel_ccc / use_dynamic_fusion / use_linear_gva / use_semantic_guided_fusion
    #   dyn_neighbors / bn_eps / bn_momentum / max_points / dropout 等
    base_dim = 64
    max_points = MAX_POINTS             # 直接与上面保持一致
    dropout_p = 0.3
    droppath_prob = 0.10
    drop_rgb_p = 0.0
    drop_normal_p = 0.0
    logit_temp = TEMP_FACTOR

    # 模块总开关
    use_channel_ccc = True              # == ATTN_USE_CAA
    use_dynamic_fusion = True           # == 动态邻域 + 通道融合
    use_linear_gva = True               # == 线性注意力（替代 softmax）
    use_semantic_guided_fusion = True   # == 语义引导全尺度融合

    # 动态邻域参数
    dyn_neighbors = 16                  # == KNN_K_BASE
    dyn_tau = 0.2

    # BN/数值
    bn_eps = 1e-3
    bn_momentum = 0.01

    # 轻几何增强开关（sgdat 内部几何嵌入）
    use_geom_enhance = True

    # ================== 日志与保存 ==================
    LOG_DIR = f"logs/{_TS}"
    MODEL_SAVE_DIR = f"checkpoints/{_TS}"
    LOG_FREQ = 10

    # ================== 随机性 ==================
    SEED = 42
    DETERMINISTIC = False

    IN_DIM = 9
    GVA_LIN_EMBED = 64
    DEBUG = False

    def __init__(self):
        # 统一种子 & 可重复性
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

        # 目录
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)

    def print_config(self):
        print("==================================================")
        print("训练配置:")
        for k, v in vars(self).items():
            if k.startswith("_") or callable(v):
                continue
            print(f"{k}: {v}")
        print("==================================================")
