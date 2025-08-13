import os
import datetime

class Config:
    def __init__(self):
        # 数据路径
        self.DATA_ROOT = "/root/my/point/data/tower"
        self.TRAIN_DIR = os.path.join(self.DATA_ROOT, "train")
        self.VAL_DIR = os.path.join(self.DATA_ROOT, "val")

        # 模型与训练参数
        self.NUM_CLASSES = 3
        self.BATCH_SIZE = 1
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.MAX_EPOCHS = 100
        self.EVAL_FREQ = 5
        self.SAVE_FREQ = 10
        self.DEVICE = "cuda"

        # 新增参数：是否限制每个样本的点数
        self.LIMIT_POINTS = False
        self.MAX_POINTS = 8000  # 仅当 LIMIT_POINTS=True 时生效

        # 数据增强
        self.ROTATION_AUG = True
        self.SCALE_AUG = True
        self.NOISE_AUG = True
        self.TRANSFORM_AUG = True

        # 训练辅助
        self.LOG_FREQ = 10
        self.CLIP_NORM = 2.0
        self.NUM_WORKERS = 4

        # 日志 & 模型保存路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.LOG_DIR = os.path.join("logs", timestamp)
        self.MODEL_SAVE_DIR = os.path.join("checkpoints", timestamp)

    def print_config(self):
        print("=" * 50)
        print("训练配置:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("=" * 50)
