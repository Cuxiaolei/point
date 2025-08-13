import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_classes, transform_aug=False, rotation_aug=False, scale_aug=False,
                 noise_aug=False, limit_points=True, max_points=20000):
        self.root_dir = root_dir
        self.scene_list = [os.path.join(root_dir, s) for s in sorted(os.listdir(root_dir))]
        self.num_classes = num_classes
        self.transform_aug = transform_aug
        self.rotation_aug = rotation_aug
        self.scale_aug = scale_aug
        self.noise_aug = noise_aug
        self.limit_points = limit_points
        self.max_points = max_points

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_path = self.scene_list[idx]
        points = np.load(os.path.join(scene_path, "points.npy"))  # shape (N, C)
        labels = np.load(os.path.join(scene_path, "labels.npy"))  # shape (N,)

        # 数据增强
        if self.transform_aug:
            points = self.random_transform(points)
        if self.rotation_aug:
            points = self.random_rotation(points)
        if self.scale_aug:
            points = self.random_scale(points)
        if self.noise_aug:
            points = self.random_noise(points)

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        # 如果限制点数，则裁剪或填充
        if self.limit_points:
            if points.shape[0] > self.max_points:
                points = points[:self.max_points]
                labels = labels[:self.max_points]
            elif points.shape[0] < self.max_points:
                pad_size = self.max_points - points.shape[0]
                points = torch.cat([points, torch.zeros(pad_size, points.shape[1])], dim=0)
                labels = torch.cat([labels, torch.full((pad_size,), -1, dtype=torch.long)], dim=0)

        return points, labels

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
