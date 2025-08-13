import os
import numpy as np
import torch
from torch.utils.data import Dataset


def random_rotation(points):
    # 仅绕 z 轴旋转（塔类场景常见）
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    pts = points.copy()
    pts[:, :3] = pts[:, :3] @ R.T
    pts[:, 6:9] = pts[:, 6:9] @ R.T  # 法向也旋转
    return pts


def random_scale(points, scale_low=0.9, scale_high=1.1):
    s = np.random.uniform(scale_low, scale_high)
    pts = points.copy()
    pts[:, :3] *= s
    return pts


def random_translate(points, max_shift=0.02):
    shift = np.random.uniform(-max_shift, max_shift, size=(1, 3)).astype(np.float32)
    pts = points.copy()
    pts[:, :3] += shift
    return pts


def add_noise(points, sigma=0.002, clip=0.01):
    jitter = np.clip(sigma * np.random.randn(points.shape[0], 3), -clip, clip).astype(np.float32)
    pts = points.copy()
    pts[:, :3] += jitter
    return pts


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
                 limit_points=True, max_points=20000):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.scene_list = [os.path.join(self.root_dir, s) for s in sorted(os.listdir(self.root_dir))]
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate
        self.limit_points = limit_points
        self.max_points = max_points

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, coord):
        # 简单增强：旋转/缩放/平移/噪声，与之前保持一致
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

    def __getitem__(self, idx):
        scene = self.scene_list[idx]
        coord = np.load(os.path.join(scene, "coord.npy")).astype(np.float32)    # (N,3)
        color = np.load(os.path.join(scene, "color.npy")).astype(np.float32)    # (N,3)
        normal = np.load(os.path.join(scene, "normal.npy")).astype(np.float32)  # (N,3)
        label = np.load(os.path.join(scene, "segment20.npy")).astype(np.int64)  # (N,)

        if self.rotation or self.scale or self.noise or self.translate:
            coord = self._augment(coord)

        pts = np.concatenate([coord, color, normal], axis=1).astype(np.float32)  # (N,9)

        pts = torch.from_numpy(pts)      # (N,9)
        lbl = torch.from_numpy(label)    # (N,)
        return pts, lbl