import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train",
                 rotation=False, scale=False, noise=False, translate=False,
                 limit_points=True, max_points=20000):
        """
        :param root_dir: 数据集根目录
        :param split: "train" 或 "val"
        :param rotation: 是否旋转增强
        :param scale: 是否缩放增强
        :param noise: 是否加噪声
        :param translate: 是否平移增强
        :param limit_points: 是否限制点数
        :param max_points: 限制的最大点数
        """
        self.root_dir = root_dir
        self.split = split
        self.scene_list = [
            os.path.join(root_dir, split, s)
            for s in sorted(os.listdir(os.path.join(root_dir, split)))
        ]
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate
        self.limit_points = limit_points
        self.max_points = max_points

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_path = self.scene_list[idx]
        points_path = os.path.join(scene_path, "points.npy")

        # 1. 优先使用 points.npy
        if os.path.exists(points_path):
            points = np.load(points_path)
        else:
            # 2. 动态拼接 coord + color + normal
            coord = np.load(os.path.join(scene_path, "coord.npy"))  # (N,3)
            color = np.load(os.path.join(scene_path, "color.npy"))  # (N,3)
            normal = np.load(os.path.join(scene_path, "normal.npy"))  # (N,3)

            # 颜色归一化到 0~1
            if color.max() > 1.0:
                color = color.astype(np.float32) / 255.0

            points = np.concatenate([coord, color, normal], axis=1)  # (N, 9)

        labels = np.load(os.path.join(scene_path, "segment20.npy"))  # (N,)

        # 数据增强
        if self.rotation:
            theta = np.random.uniform(0, 2 * np.pi)
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]], dtype=np.float32)
            points[:, :3] = points[:, :3] @ rot_matrix

        if self.scale:
            scale_factor = np.random.uniform(0.9, 1.1)
            points[:, :3] *= scale_factor

        if self.noise:
            noise = np.random.normal(0, 0.005, size=points[:, :3].shape)
            points[:, :3] += noise

        if self.translate:
            translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
            points[:, :3] += translation

        # 限制点数
        if self.limit_points and self.max_points is not None and points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[choice]
            labels = labels[choice]

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def random_transform(self, pts):
        # 占位：这里可以加额外的空间变换
        return pts

    def random_rotation(self, pts):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
        pts[:, :3] = pts[:, :3] @ rot_matrix
        return pts

    def random_scale(self, pts, scale_range=(0.9, 1.1)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        pts[:, :3] *= scale
        return pts

    def random_noise(self, pts, sigma=0.005):
        noise = np.random.normal(0, sigma, pts[:, :3].shape)
        pts[:, :3] += noise
        return pts
