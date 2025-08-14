import os
import numpy as np
import laspy
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import random
import json

np.set_printoptions(suppress=True, precision=8)

# 三分类映射：0=塔类，1=背景类，2=导线类
CATEGORY_TO_SEGMENT = {
    "铁塔": 0,
    "绝缘子": 0,
    "建筑物点": 1,
    "公路": 1,
    "低点": 1,
    "地面点": 1,
    "变电站": 1,
    "中等植被点": 1,
    "导线": 2,
    "引流线": 2,
    "地线": 2,
}

def group_files_by_scene(input_dir):
    """按文件名前缀分组，如 1-2(1_2)_变电站.las -> 场景1-2"""
    scene_groups = {}
    for las_file in Path(input_dir).glob("*.las"):
        stem = las_file.stem
        if "(" in stem:
            scene_id = stem.split("(")[0]
        else:
            scene_id = "_".join(stem.split("_")[:-1])
        scene_groups.setdefault(scene_id, []).append(las_file)
    return scene_groups

def estimate_normals(coords, k=15):
    """在原始尺度下计算法向量"""
    if len(coords) == 0:
        return np.zeros_like(coords)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals)

def main(input_dir, output_dir, ratio=0.8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    scene_groups = group_files_by_scene(input_dir)

    for idx, (scene_id, las_files) in enumerate(scene_groups.items()):
        scene_name = f"scene{idx:04d}_00"
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(exist_ok=True)

        all_coords = []
        all_colors = []
        all_normals = []
        all_labels = []

        print(f"处理场景 {scene_id} 包含文件数: {len(las_files)}")

        for las_file in las_files:
            name = las_file.stem
            class_name = name.split("_")[-1]
            if class_name not in CATEGORY_TO_SEGMENT:
                print(f"跳过未知类别: {class_name}")
                continue

            label_id = CATEGORY_TO_SEGMENT[class_name]

            las = laspy.read(str(las_file))
            coords = np.vstack((las.x, las.y, las.z)).T

            # 法向量计算在原始坐标系
            normals = estimate_normals(coords)

            # 颜色
            colors = np.vstack((las.red, las.green, las.blue)).T
            colors = (colors / 65535.0 * 255).astype(np.uint8)

            labels = np.full(coords.shape[0], label_id, dtype=np.uint8)

            all_coords.append(coords.astype(np.float32))
            all_colors.append(colors)
            all_normals.append(normals.astype(np.float32))
            all_labels.append(labels)

        # 合并
        all_coords = np.concatenate(all_coords, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 下采样类别 1 和 2
        base_count = np.sum(all_labels == 0)
        target_1 = int(base_count * ratio)
        target_2 = int(base_count * ratio)

        indices_0 = np.where(all_labels == 0)[0]
        indices_1 = np.where(all_labels == 1)[0]
        indices_2 = np.where(all_labels == 2)[0]

        sampled_1 = np.random.choice(indices_1, size=min(len(indices_1), target_1), replace=False)
        sampled_2 = np.random.choice(indices_2, size=min(len(indices_2), target_2), replace=False)

        final_indices = np.concatenate([indices_0, sampled_1, sampled_2])
        np.random.shuffle(final_indices)

        # 应用采样
        all_coords = all_coords[final_indices]
        all_colors = all_colors[final_indices]
        all_normals = all_normals[final_indices]
        all_labels = all_labels[final_indices]

        print(f"[{scene_name}] 类别分布: 0={np.sum(all_labels==0)}, 1={np.sum(all_labels==1)}, 2={np.sum(all_labels==2)}")

        # 保存（坐标先不归一化，归一化放到 data_split 里统一做）
        np.save(scene_dir / "coord.npy", all_coords)
        np.save(scene_dir / "color.npy", all_colors)
        np.save(scene_dir / "normal.npy", all_normals)
        np.save(scene_dir / "label.npy", all_labels)

if __name__ == "__main__":
    ratio = 0.8
    input_dir = r"D:/user/code/AI/Point++/data/input"
    output_dir = r"D:/user/code/AI/Point++/data/output"
    main(input_dir, output_dir, ratio)
