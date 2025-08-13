import os
import numpy as np
import torch
from torch.utils.data import Dataset
from math import isfinite

class PointCloudDataset(Dataset):
    """
    每个场景目录下：
      - coord.npy   (N,3)
      - color.npy   (N,3)
      - normal.npy  (N,3)
      - segment20.npy (N,)  # label, 取值 {0..K-1}，无 pad；pad 在 collate 里补 -1
    """
    def __init__(self, root_dir, split="train",
                 rotation=False, scale=False, noise=False, translate=False,
                 limit_points=True, max_points=20000, normalize=True, num_classes=None):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.scene_list = [os.path.join(self.root_dir, s) for s in sorted(os.listdir(self.root_dir))]
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate
        self.limit_points = limit_points
        self.max_points = max_points
        self.normalize = normalize
        self.num_classes = num_classes  # 用于检查标签合法性

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, coord):
        xyz = coord.copy()
        if self.rotation:
            theta = np.random.uniform(0, 2*np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            R = np.array([[cos, -sin, 0],[sin, cos, 0],[0,0,1]], dtype=np.float32)
            xyz = (xyz @ R.T).astype(np.float32)
        if self.scale:
            s = np.random.uniform(0.9, 1.1)
            xyz = (xyz * s).astype(np.float32)
        if self.translate:
            t = np.random.uniform(-0.1, 0.1, size=(1,3)).astype(np.float32)
            xyz = xyz + t
        if self.noise:
            n = np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
            xyz = xyz + n
        return xyz

    def _normalize_coords(self, coords):
        # 平移到中心
        center = np.mean(coords, axis=0, keepdims=True)
        coords -= center
        # 缩放到单位球
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist
        return coords.astype(np.float32)

    def __getitem__(self, idx):
        scene = self.scene_list[idx]
        coord = np.load(os.path.join(scene, "coord.npy")).astype(np.float32)    # (N,3)
        color = np.load(os.path.join(scene, "color.npy")).astype(np.float32)    # (N,3)
        normal = np.load(os.path.join(scene, "normal.npy")).astype(np.float32)  # (N,3)
        label = np.load(os.path.join(scene, "segment20.npy")).astype(np.int64)  # (N,)

        # 标签合法性检查
        if self.num_classes is not None:
            assert np.all((label == -1) | ((label >= 0) & (label < self.num_classes))), \
                f"标签值超出范围: {np.unique(label)}"

        # 数据增强
        if self.rotation or self.scale or self.noise or self.translate:
            coord = self._augment(coord)

        # 归一化
        if self.normalize:
            coord = self._normalize_coords(coord)

        # 组装特征
        pts = np.concatenate([coord, color, normal], axis=1).astype(np.float32)  # (N,9)

        # 限制点数
        if self.limit_points and pts.shape[0] > self.max_points:
            choice = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[choice]
            label = label[choice]

        pts = torch.from_numpy(pts).float()  # 保证 float32
        lbl = torch.from_numpy(label).long() # 保证 int64
        return pts, lbl
