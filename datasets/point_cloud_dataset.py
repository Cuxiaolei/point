import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class PointCloudDataset(Dataset):
    """点云数据集加载器（优化版）"""

    def __init__(self, root_dir, split="train", transform=None, max_points=20000):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集分割类型 ("train" 或 "val")
            transform: 数据增强变换
            max_points: 单一场景的最大点数量（核心优化）
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_points = max_points  # 限制单场景最大点数，解决内存溢出
        self.scene_list = self._get_scene_list()

        # 打印数据集信息
        print(f"加载 {split} 数据集: {self.root_dir}")
        print(f"包含 {len(self.scene_list)} 个场景，最大点数限制: {max_points}")

    def _get_scene_list(self):
        """获取场景列表，过滤无效场景"""
        scene_list = []
        for scene in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_path) and self._has_valid_files(scene_path):
                scene_list.append(scene_path)
        return scene_list

    def _has_valid_files(self, scene_path):
        """检查场景目录是否包含有效的点云文件"""
        required_files = ["coord.npy", "segment20.npy"]
        for file in required_files:
            file_path = os.path.join(scene_path, file)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False
        return True

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        """加载单个场景的数据（优化版）"""
        scene_path = self.scene_list[idx]

        # 加载点云数据
        points = np.load(os.path.join(scene_path, "coord.npy"))  # (N, 3)
        labels = np.load(os.path.join(scene_path, "segment20.npy"))  # (N,)

        # 核心优化1：限制最大点数（超过则下采样，不足则补零）
        if len(points) > self.max_points:
            # 随机下采样（保留均匀分布的点）
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            labels = labels[indices]
        elif len(points) < self.max_points:
            # 不足则补零（保持批次形状一致，避免动态形状）
            pad_size = self.max_points - len(points)
            points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant')
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=-1)  # 用-1标记无效点

        # 可选：加载颜色和法向量特征（如果有）
        features = []
        # 加载颜色特征（严格检查点数一致性）
        if os.path.exists(os.path.join(scene_path, "color.npy")):
            colors = np.load(os.path.join(scene_path, "color.npy"))
            # 强制检查：颜色点数必须与点云点数一致，否则丢弃
            if len(colors) != len(points):
                print(
                    f"警告：场景 {scene_path} 的color.npy点数与点云不一致（{len(colors)} vs {len(points)}），已丢弃颜色特征")
            else:
                # 同步下采样/补零
                if len(points) > self.max_points and indices is not None:
                    colors = colors[indices]
                elif len(points) < self.max_points:
                    colors = np.pad(colors, ((0, pad_size), (0, 0)), mode='constant')
                features.append(colors)

        # 加载法向量特征（同上检查）
        if os.path.exists(os.path.join(scene_path, "normal.npy")):
            normals = np.load(os.path.join(scene_path, "normal.npy"))
            if len(normals) != len(points):
                print(
                    f"警告：场景 {scene_path} 的normal.npy点数与点云不一致（{len(normals)} vs {len(points)}），已丢弃法向量特征")
            else:
                # 同步下采样/补零
                if len(points) > self.max_points and indices is not None:
                    normals = normals[indices]
                elif len(points) < self.max_points:
                    normals = np.pad(normals, ((0, pad_size), (0, 0)), mode='constant')
                features.append(normals)

        # 合并特征（如果有）
        if features:
            points = np.concatenate([points] + features, axis=-1)  # (N, 3 + C)

        # 转换为张量
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        # 应用数据增强（优化：只对有效点增强）
        if self.transform:
            # 分离有效点和无效点（补零的部分不参与增强）
            valid_mask = labels != -1
            valid_points = points[valid_mask]
            valid_labels = labels[valid_mask]

            # 只对有效点进行增强
            valid_points, valid_labels = self.transform(valid_points, valid_labels)

            # 重建完整点云和标签
            points[valid_mask] = valid_points
            labels[valid_mask] = valid_labels

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