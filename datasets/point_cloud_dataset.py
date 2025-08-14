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

    新增适配项：
      - color_mode: {"auto","01","255"}，默认 "auto"：若检测到颜色最大值>1，则 /255 归一化
      - normal_unit: 是否对 normal 每行做单位化
    """

    def __init__(self, data_root, split="train",
                 rotation=False, scale=False, noise=False, translate=False,
                 limit_points=True, max_points=20000, normalize=True, num_classes=None,
                 color_mode="auto", normal_unit=True,
                 # ======= 新增增强参数（保持向后兼容） =======
                 point_dropout=False,
                 point_dropout_rate=0.1,
                 cutmix=False,
                 cutmix_prob=0.0,
                 cutmix_ratio=0.15,
                 color_jitter=False,
                 color_jitter_params=None,
                 normal_noise_std=0.0):
        super().__init__()
        self.root_dir = os.path.join(data_root, split)
        self.scene_list = [os.path.join(self.root_dir, s) for s in sorted(os.listdir(self.root_dir))]
        self.rotation = rotation
        self.scale = scale
        self.noise = noise
        self.translate = translate
        self.limit_points = limit_points
        self.max_points = max_points if (limit_points and max_points is not None) else None
        self.normalize = normalize
        self.num_classes = num_classes  # 用于检查标签合法性

        # 新增：值域/单位化选项
        self.color_mode = color_mode if color_mode in {"auto", "01", "255"} else "auto"
        self.normal_unit = bool(normal_unit)

        # ======= 新增增强选项的保存 =======
        self.split = split
        self.point_dropout = bool(point_dropout)
        self.point_dropout_rate = float(point_dropout_rate)
        self.cutmix = bool(cutmix)
        self.cutmix_prob = float(cutmix_prob)
        self.cutmix_ratio = float(cutmix_ratio)
        self.color_jitter = bool(color_jitter)
        self.color_jitter_params = color_jitter_params if color_jitter_params is not None else dict(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
        )
        self.normal_noise_std = float(normal_noise_std)

    def __len__(self):
        return len(self.scene_list)

    # ============== 你原有的几何增强逻辑（保持不变） ==============
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

    def _normalize_color(self, color):
        if self.color_mode == "255":
            return color.astype(np.float32)
        if self.color_mode == "01":
            # 假设传入 0~255，做 /255
            return (color.astype(np.float32) / 255.0).astype(np.float32)
        # auto：如果出现 >1 的值，视为 0-255，做 /255
        if np.max(color) > 1.0:
            return (color.astype(np.float32) / 255.0).astype(np.float32)
        return color.astype(np.float32)

    def _unit_normal(self, normal):
        if not self.normal_unit:
            return normal.astype(np.float32)
        n = normal.astype(np.float32)
        norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = n / norm
        return n.astype(np.float32)

    # ===================== 新增：Point Dropout =====================
    def _apply_point_dropout(self, coord, color, normal, label):
        """按比例丢弃随机点，并用重复点补回长度，保持 N 不变"""
        N = coord.shape[0]
        drop_num = int(N * max(0.0, min(1.0, self.point_dropout_rate)))
        if drop_num <= 0 or drop_num >= N:
            return coord, color, normal, label
        drop_idx = np.random.choice(N, drop_num, replace=False)
        keep_mask = np.ones(N, dtype=bool)
        keep_mask[drop_idx] = False
        # 保留
        c1, c2, n1, l1 = coord[keep_mask], color[keep_mask], normal[keep_mask], label[keep_mask]
        # 补齐
        pad_idx = np.random.choice(c1.shape[0], drop_num, replace=True)
        coord_new = np.concatenate([c1, c1[pad_idx]], axis=0)
        color_new = np.concatenate([c2, c2[pad_idx]], axis=0)
        normal_new = np.concatenate([n1, n1[pad_idx]], axis=0)
        label_new = np.concatenate([l1, l1[pad_idx]], axis=0)
        return coord_new, color_new, normal_new, label_new

    # ===================== 新增：PointCutMix =====================
    def _apply_pointcutmix(self, coord, color, normal, label):
        """以概率执行点级替换，从另一随机样本替换比例为 cutmix_ratio 的点"""
        if np.random.rand() > self.cutmix_prob or len(self.scene_list) <= 1:
            return coord, color, normal, label
        # 随机找一个不同的样本
        ridx = np.random.randint(0, len(self.scene_list))
        other_path = self.scene_list[ridx]
        try:
            o_coord = np.load(os.path.join(other_path, "coord.npy")).astype(np.float32)
            o_color = np.load(os.path.join(other_path, "color.npy")).astype(np.float32)
            o_normal = np.load(os.path.join(other_path, "normal.npy")).astype(np.float32)
            o_label = np.load(os.path.join(other_path, "segment20.npy")).astype(np.int64)
        except Exception:
            return coord, color, normal, label

        N = coord.shape[0]
        M = min(N, o_coord.shape[0])
        if M <= 1:
            return coord, color, normal, label
        num_replace = int(M * max(0.0, min(1.0, self.cutmix_ratio)))
        if num_replace <= 0:
            return coord, color, normal, label
        # 在交集长度内做替换
        replace_idx = np.random.choice(M, num_replace, replace=False)
        coord[:M][replace_idx] = o_coord[:M][replace_idx]
        color[:M][replace_idx] = o_color[:M][replace_idx]
        normal[:M][replace_idx] = o_normal[:M][replace_idx]
        label[:M][replace_idx] = o_label[:M][replace_idx]
        return coord, color, normal, label

    # ===================== 新增：颜色抖动 & 法向扰动 =====================
    def _apply_color_jitter_and_normal_noise(self, color, normal):
        c = color.astype(np.float32)
        n = normal.astype(np.float32)

        # 颜色抖动（简单、尺度无关）：仅在训练启用
        if self.color_jitter:
            # 统一确定值域上限，避免 255 模式越界
            vmax = 255.0 if np.max(c) > 1.0 else 1.0
            p = self.color_jitter_params
            b = float(p.get("brightness", 0.0))
            ct = float(p.get("contrast", 0.0))
            st = float(p.get("saturation", 0.0))
            # brightness：整体乘法
            if b > 0:
                c = c * np.random.uniform(1.0 - b, 1.0 + b)
            # contrast：减均值后缩放
            if ct > 0:
                mean = np.mean(c, axis=1, keepdims=True)
                c = (c - mean) * np.random.uniform(1.0 - ct, 1.0 + ct) + mean
            # saturation：向灰度靠拢/远离
            if st > 0:
                gray = np.mean(c, axis=1, keepdims=True)
                c = gray + (c - gray) * np.random.uniform(1.0 - st, 1.0 + st)
            c = np.clip(c, 0.0, vmax)

        # 法向扰动（高斯）+ 再单位化：仅在训练启用
        if self.normal_noise_std > 0.0:
            n = n + np.random.normal(0.0, self.normal_noise_std, size=n.shape).astype(np.float32)
            norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            n = n / norm

        return c.astype(np.float32), n.astype(np.float32)

    # =============================== 取样 ===============================
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

        # ===================== 仅在训练集执行：CutMix → Dropout =====================
        if self.split == "train":
            if self.cutmix:
                coord, color, normal, label = self._apply_pointcutmix(coord, color, normal, label)
            if self.point_dropout:
                coord, color, normal, label = self._apply_point_dropout(coord, color, normal, label)

        # 数据增强（你原有的几何增强）
        if self.rotation or self.scale or self.noise or self.translate:
            coord = self._augment(coord)

        # 归一化（保持你原有的三个函数调用）
        if self.normalize:
            coord = self._normalize_coords(coord)
        color = self._normalize_color(color)
        normal = self._unit_normal(normal)

        # 颜色抖动 & 法向扰动（仅 train 时）
        if self.split == "train" and (self.color_jitter or self.normal_noise_std > 0.0):
            color, normal = self._apply_color_jitter_and_normal_noise(color, normal)

        # 组装特征
        pts = np.concatenate([coord, color, normal], axis=1).astype(np.float32)  # (N,9)

        # 限制点数
        if self.limit_points and self.max_points is not None and pts.shape[0] > self.max_points:
            choice = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[choice]
            label = label[choice]

        pts = torch.from_numpy(pts).float()  # 保证 float32
        lbl = torch.from_numpy(label).long() # 保证 int64

        print(f"[DEBUG] Dataset output points.shape = {pts.shape}")
        print(f"[DEBUG] First point (xyz/rgb/normal/...): {pts[0]}")
        return pts, lbl
