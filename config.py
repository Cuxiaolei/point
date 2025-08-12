import os
import torch  # 新增导入，用于判断CUDA是否可用
from datetime import datetime


class Config:
    """模型训练和评估的配置参数（Linux系统适配版）"""

    # 数据集配置（Linux路径格式，使用/分隔）
    DATA_ROOT = "/root/my/Point++/data/tower"  # 划分后的数据集根目录（根据实际路径调整）
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")  # 训练集目录
    VAL_DIR = os.path.join(DATA_ROOT, "val")  # 验证集目录
    NUM_CLASSES = 13  # 类别数量（根据实际数据集调整）

    # 训练配置
    BATCH_SIZE = 1  # 批次大小（根据GPU显存调整）
    LEARNING_RATE = 1e-3  # 初始学习率
    WEIGHT_DECAY = 1e-5  # 权重衰减
    MAX_EPOCHS = 100  # 最大训练轮数
    MAX_POINTS = 8000  # 从20000减少到8000（可根据GPU显存调整）
    # ... 其他已有配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 修复CUDA判断，使用torch

    # 数据增强配置
    ROTATION_AUG = True  # 是否启用旋转增强
    SCALE_AUG = True  # 是否启用尺度增强
    NOISE_AUG = True  # 是否启用噪声添加
    TRANSFORM_AUG = True  # 是否启用平移增强

    # 模型配置（Linux路径）
    MODEL_SAVE_DIR = os.path.join(
        "checkpoints",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )  # 模型保存目录
    SAVE_FREQ = 10  # 模型保存频率（每多少轮）

    # 日志配置（Linux路径）
    LOG_DIR = os.path.join(
        "logs",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )  # 日志保存目录
    LOG_FREQ = 10  # 日志打印频率（每多少个批次）

    # 评估配置
    EVAL_FREQ = 5  # 评估频率（每多少轮）

    def __init__(self, **kwargs):
        """允许通过关键字参数修改配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"配置中不存在属性: {key}")

    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("训练配置:")
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                print(f"{attr}: {getattr(self, attr)}")
        print("=" * 50)
