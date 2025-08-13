# point_cloud_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, max_points=20000):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_points = max_points
        self.scene_list = self._get_scene_list()

    def _get_scene_list(self):
        scene_list = []
        for scene in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_path) and self._has_valid_files(scene_path):
                scene_list.append(scene_path)
        return scene_list

    def _has_valid_files(self, scene_path):
        # 必须包含 coord、color、normal、segment 文件
        required_files = ["coord.npy", "color.npy", "normal.npy", "segment20.npy"]
        for file in required_files:
            file_path = os.path.join(scene_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False
        return True

    def __getitem__(self, idx):
        scene_path = self.scene_list[idx]

        coords = np.load(os.path.join(scene_path, "coord.npy"))    # (N, 3)
        colors = np.load(os.path.join(scene_path, "color.npy"))    # (N, 3)
        normals = np.load(os.path.join(scene_path, "normal.npy"))  # (N, 3)
        labels = np.load(os.path.join(scene_path, "segment20.npy"))# (N,)

        # 如果点云数据过多，采样
        if len(coords) > self.max_points:
            indices = np.random.choice(len(coords), self.max_points, replace=False)
            coords = coords[indices]
            colors = colors[indices]
            normals = normals[indices]
            labels = labels[indices]
        elif len(coords) < self.max_points:
            pad_size = self.max_points - len(coords)
            coords = np.pad(coords, ((0, pad_size), (0, 0)), mode='constant')
            colors = np.pad(colors, ((0, pad_size), (0, 0)), mode='constant')
            normals = np.pad(normals, ((0, pad_size), (0, 0)), mode='constant')
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=-1)

        # 归一化颜色到 [0, 1]，拼接成 9 维特征 (XYZ + RGB + Normal)
        points = np.concatenate([coords, colors / 255.0, normals], axis=-1)
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        if self.transform:
            points, labels = self.transform(points, labels)

        return points, labels

    def __len__(self):
        return len(self.scene_list)


class PointCloudTransform:
    """点云数据增强"""
    def __init__(self, rotation=True, scale=True, noise=True, translate=True):
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate

    def __call__(self, points, labels):
        if len(points) == 0:
            return points, labels

        coords = points[:, :3]
        features = points[:, 3:]

        if self.rotation:
            coords = self._random_rotation(coords)
        if self.scale:
            coords = self._random_scale(coords)
        if self.translate:
            coords = self._random_translate(coords)
        if self.noise:
            coords = self._add_noise(coords)

        points = torch.cat([coords, features], dim=-1)
        return points, labels

    def _random_rotation(self, coords):
        angle = torch.rand(1) * np.pi / 3 - np.pi / 6
        rot_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ], dtype=coords.dtype)
        return coords @ rot_matrix

    def _random_scale(self, coords):
        scale = torch.rand(1) * 0.4 + 0.8
        return coords * scale

    def _random_translate(self, coords):
        translate = torch.randn(3) * 0.03
        return coords + translate

    def _add_noise(self, coords):
        noise = torch.randn_like(coords) * 0.0005
        return coords + noise
