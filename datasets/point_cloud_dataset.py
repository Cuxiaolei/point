import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class PointCloudDataset(Dataset):
    """点云数据集加载器（优化版）"""
    def __init__(self, root_dir, split="train", transform=None, max_points=20000):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_points = max_points  # 限制单场景最大点数，解决内存溢出
        self.scene_list = self._get_scene_list()

    def _get_scene_list(self):
        scene_list = []
        for scene in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_path) and self._has_valid_files(scene_path):
                scene_list.append(scene_path)
        return scene_list

    def _has_valid_files(self, scene_path):
        required_files = ["coord.npy", "segment20.npy"]
        for file in required_files:
            file_path = os.path.join(scene_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False
        return True

    def __getitem__(self, idx):
        scene_path = self.scene_list[idx]
        points = np.load(os.path.join(scene_path, "coord.npy"))  # (N, 3)
        labels = np.load(os.path.join(scene_path, "segment20.npy"))  # (N,)

        # 输出数据维度用于调试
        print(f"[PointCloudDataset] Loaded points shape: {points.shape}, labels shape: {labels.shape}")

        if os.path.exists(os.path.join(scene_path, "color.npy")):
            colors = np.load(os.path.join(scene_path, "color.npy"))
            if len(colors) != len(points):
                colors = np.pad(colors, ((0, len(points) - len(colors)), (0, 0)), mode='constant')
        else:
            colors = np.zeros_like(points)  # 默认颜色

        if os.path.exists(os.path.join(scene_path, "normal.npy")):
            normals = np.load(os.path.join(scene_path, "normal.npy"))
            if len(normals) != len(points):
                normals = np.pad(normals, ((0, len(points) - len(normals)), (0, 0)), mode='constant')
        else:
            normals = np.zeros_like(points)  # 默认法向量

        # 核心优化：限制最大点数（超过则下采样，不足则补零）
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            labels = labels[indices]
            colors = colors[indices]
            normals = normals[indices]
        elif len(points) < self.max_points:
            pad_size = self.max_points - len(points)
            points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant')
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=-1)
            colors = np.pad(colors, ((0, pad_size), (0, 0)), mode='constant')
            normals = np.pad(normals, ((0, pad_size), (0, 0)), mode='constant')

        # 合并特征（如果有）
        points = torch.from_numpy(np.concatenate([points, colors, normals], axis=-1)).float()
        labels = torch.from_numpy(labels).long()

        # 如果有数据增强操作，则应用
        if self.transform:
            valid_mask = labels != -1
            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]

            valid_points, valid_labels = self.transform(valid_points, valid_labels)

            points[valid_mask] = valid_points
            labels[valid_mask] = valid_labels

        print(f"[PointCloudDataset] Final points shape: {points.shape}, labels shape: {labels.shape}")
        return points, labels

class PointCloudTransform:
    """点云数据增强变换（优化版）"""

    def __init__(self, rotation=True, scale=True, noise=True, translate=True):
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate

    def __call__(self, points, labels):
        """
        对输入点云进行数据增强（只处理有效点）

        Args:
            points: (N, 3 + C) 点云坐标和特征（仅有效点）
            labels: (N,) 点云标签（仅有效点）

        Returns:
            增强后的点云和标签
        """
        if len(points) == 0:  # 避免空场景报错
            return points, labels

        # 分离坐标和特征
        coords = points[:, :3]
        features = points[:, 3:] if points.shape[1] > 3 else None

        # 随机旋转（只旋转坐标）
        if self.rotation:
            coords = self._random_rotation(coords)

        # 随机缩放
        if self.scale:
            coords = self._random_scale(coords)

        # 随机平移（优化：缩小平移范围，避免过度偏移）
        if self.translate:
            coords = self._random_translate(coords)

        # 添加高斯噪声（优化：降低噪声幅度）
        if self.noise:
            coords = self._add_noise(coords)

        # 重新组合坐标和特征
        if features is not None:
            points = torch.cat([coords, features], dim=-1)
        else:
            points = coords

        return points, labels

    def _random_rotation(self, coords):
        """绕Z轴随机旋转（优化：限制旋转角度范围）"""
        angle = torch.rand(1) * np.pi / 3 - np.pi / 6  # 限制在±30度，避免过度旋转
        rot_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ], dtype=coords.dtype)
        return coords @ rot_matrix

    def _random_scale(self, coords):
        """随机缩放（保持原有逻辑）"""
        scale = torch.rand(1) * 0.4 + 0.8  # 0.8-1.2之间的随机缩放因子
        return coords * scale

    def _random_translate(self, coords):
        """随机平移（优化：缩小平移范围）"""
        translate = torch.randn(3) * 0.03  # 从0.05缩小到0.03，减少偏移量
        return coords + translate

    def _add_noise(self, coords):
        """添加高斯噪声（优化：降低噪声幅度）"""
        noise = torch.randn_like(coords) * 0.0005  # 从0.001减半，减少噪声影响
        return coords + noise