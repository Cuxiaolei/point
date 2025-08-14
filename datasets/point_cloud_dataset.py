import os
import numpy as np
import torch
from torch.utils.data import Dataset
from math import isfinite

class PointCloudDataset(Dataset):
    """
    每个场景目录下：
      - coord.npy     (N,3)
      - color.npy     (N,3)
      - normal.npy    (N,3)
      - segment20.npy (N,)  # label, 取值 {0..K-1}，无 pad；pad 在 collate 里补 -1

    适配项：
      - color_mode: {"auto","01","255"}，默认 "auto"：若检测到颜色最大值>1，则 /255 归一化
      - normal_unit: 是否对 normal 每行做单位化

    新增增强（仅 train 时启用）：
      - 随机旋转/缩放/平移/噪声
      - 各向异性缩放（更贴近真实形变）
      - 颜色抖动（高斯噪声）
      - 法向扰动（小角度高斯噪声后再单位化）
    """

    def __init__(self, data_root, split="train",
                 rotation=False, scale=False, noise=False, translate=False,
                 limit_points=True, max_points=20000, normalize=True, num_classes=None,
                 color_mode="auto", normal_unit=True,
                 # 新增的可控增强超参
                 aniso_scale=False, aniso_scale_range=(0.9, 1.1),
                 color_jitter_std=0.02,    # 针对 0~1 颜色值的小扰动；若是 0~255，会自动 /255 后再抖动
                 normal_jitter_std=0.03    # 法向小角度扰动幅度（弧度近似）
                 ):
        super().__init__()
        self.root_dir = os.path.join(data_root, split)
        self.scene_list = [os.path.join(self.root_dir, s) for s in sorted(os.listdir(self.root_dir))]
        self.split = split

        # 基础增强开关
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate

        # 额外增强
        self.aniso_scale = aniso_scale
        self.aniso_scale_range = aniso_scale_range
        self.color_jitter_std = float(color_jitter_std)
        self.normal_jitter_std = float(normal_jitter_std)

        self.limit_points = limit_points
        self.max_points = max_points if (limit_points and max_points is not None) else None
        self.normalize = normalize
        self.num_classes = num_classes  # 用于检查标签合法性

        # 值域/单位化选项
        self.color_mode = color_mode if color_mode in {"auto", "01", "255"} else "auto"
        self.normal_unit = bool(normal_unit)

    def __len__(self):
        return len(self.scene_list)

    # -----------------------
    # 增强与规范化工具函数
    # -----------------------
    def _augment_geom(self, coord):
        xyz = coord.copy()
        # z 轴旋转
        if self.rotation:
            theta = np.random.uniform(0, 2*np.pi)
            cos, sin = np.cos(theta), np.sin(theta)
            R = np.array([[cos, -sin, 0],[sin, cos, 0],[0,0,1]], dtype=np.float32)
            xyz = (xyz @ R.T).astype(np.float32)
        # 各向同性缩放
        if self.scale:
            s = np.random.uniform(0.9, 1.1)
            xyz = (xyz * s).astype(np.float32)
        # 各向异性缩放
        if self.aniso_scale:
            lo, hi = self.aniso_scale_range
            sx, sy, sz = np.random.uniform(lo, hi, size=3)
            S = np.diag([sx, sy, sz]).astype(np.float32)
            xyz = (xyz @ S.T).astype(np.float32)
        # 平移
        if self.translate:
            t = np.random.uniform(-0.1, 0.1, size=(1,3)).astype(np.float32)
            xyz = xyz + t
        # 高斯噪声
        if self.noise:
            n = np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
            xyz = xyz + n
        return xyz

    def _normalize_coords(self, coords):
        # 平移到中心
        center = np.mean(coords, axis=0, keepdims=True)
        coords = coords - center
        # 缩放到单位球
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist
        return coords.astype(np.float32)

    def _normalize_color(self, color):
        if self.color_mode == "255":
            return color.astype(np.float32)
        if self.color_mode == "01":
            return (color.astype(np.float32) / 255.0).astype(np.float32)
        # auto：如果出现 >1 的值，视为 0-255，做 /255
        if np.max(color) > 1.0:
            return (color.astype(np.float32) / 255.0).astype(np.float32)
        return color.astype(np.float32)

    def _unit_normal(self, normal):
        n = normal.astype(np.float32)
        norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = n / norm
        return n.astype(np.float32)

    def _color_jitter(self, color01):
        """对 0~1 颜色做高斯抖动并裁剪回 [0,1]"""
        if self.split != "train":
            return color01
        if self.color_jitter_std <= 0:
            return color01
        jitter = np.random.normal(0, self.color_jitter_std, size=color01.shape).astype(np.float32)
        c = color01 + jitter
        return np.clip(c, 0.0, 1.0).astype(np.float32)

    def _normal_jitter(self, normal):
        """对法向做小角度高斯扰动：在切平面采样一个小向量再归一化"""
        if self.split != "train":
            return normal
        if self.normal_jitter_std <= 0:
            return normal
        n = normal.astype(np.float32)
        # 生成与 n 垂直的小扰动
        rand_vec = np.random.normal(0, self.normal_jitter_std, size=n.shape).astype(np.float32)
        # 去除沿 n 的分量
        proj = (rand_vec * n).sum(axis=1, keepdims=True) * n
        t = rand_vec - proj
        n2 = n + t
        # 重新单位化
        norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
        n2 = n2 / norm
        return n2.astype(np.float32)

    # -----------------------
    # 主入口
    # -----------------------
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

        # 几何增强（仅 train）
        if self.split == "train" and (self.rotation or self.scale or self.noise or self.translate or self.aniso_scale):
            coord = self._augment_geom(coord)

        # 归一化（坐标）
        if self.normalize:
            coord = self._normalize_coords(coord)

        # 颜色与法向规范化
        color = self._normalize_color(color)   # -> 0~1
        normal = self._unit_normal(normal) if self.normal_unit else normal.astype(np.float32)

        # 颜色/法向增强（仅 train）
        color = self._color_jitter(color)
        normal = self._normal_jitter(normal) if self.normal_unit else normal

        # 组装特征
        pts = np.concatenate([coord, color, normal], axis=1).astype(np.float32)  # (N,9)

        # 限制点数（随机下采样）
        if self.limit_points and self.max_points is not None and pts.shape[0] > self.max_points:
            choice = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[choice]
            label = label[choice]

        pts = torch.from_numpy(pts).float()   # (N,9)
        lbl = torch.from_numpy(label).long()  # (N,)
        return pts, lbl
