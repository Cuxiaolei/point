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
