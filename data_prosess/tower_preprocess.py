import os
import numpy as np
import laspy
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import random
import open3d as o3d

# 设置 NumPy 打印格式：禁用科学计数法，保留 8 位小数
np.set_printoptions(suppress = True, precision = 8)

# 分类映射
CATEGORY_TO_SEGMENT = {
    "铁塔": 0,
    "绝缘子": 1,
    "建筑物点": 2,
    "公路": 3,
    "低点": 4,
    "地面点": 5,
    "变电站": 6,
    "中等植被点": 7,
    "导线": 8,
    "引流线": 9,
    "地线": 10,
}

SEGMENT_TO_LABEL = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 2
}


def group_files_by_scene(input_dir):
    """按照文件名前缀分组，如 1-2(1_2)_变电站.las -> 场景1-2"""
    scene_groups = {}
    for las_file in Path(input_dir).glob("*.las"):
        stem = las_file.stem
        if "(" in stem:
            scene_id = stem.split("(")[0]  # 提取如 1-2
        else:
            scene_id = "_".join(stem.split("_")[:-1])  # fallback
        scene_groups.setdefault(scene_id, []).append(las_file)
    return scene_groups


# 法向量估计
def estimate_normals(coords, k = 15):
    if len(coords) == 0:
        return np.zeros_like(coords)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn = k)
    )

    normals = np.asarray(pcd.normals)
    return normals


# 计算全局中心（用于场景对齐）
def compute_global_center(input_dir):
    input_dir = Path(input_dir)
    las_files = list(input_dir.glob("*.las"))
    total_sum = np.zeros(3)
    total_count = 0
    for las_file in tqdm(las_files, desc = "Computing global center"):
        las = laspy.read(str(las_file))
        coords = np.vstack((las.x, las.y, las.z)).T
        total_sum += coords.sum(axis = 0)
        total_count += coords.shape[0]
    global_center = total_sum / total_count
    print(f"Global center: {global_center}")
    return global_center


# 主处理函数
def main(input_dir, output_dir, ratio):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True)

    global_center = compute_global_center(input_dir)
    scene_groups = group_files_by_scene(input_dir)

    for idx, (scene_id, las_files) in enumerate(scene_groups.items()):
        scene_name = f"scene{idx:04d}_00"
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(exist_ok = True)

        all_coords = []
        all_colors = []
        all_normals = []
        all_segments = []

        print(f"处理场景 {scene_id} 包含文件数: {len(las_files)}")

        for las_file in las_files:
            name = las_file.stem
            class_name = name.split("_")[-1]
            if class_name not in CATEGORY_TO_SEGMENT:
                print(f"跳过未知类别: {class_name}")
                continue

            class_id = CATEGORY_TO_SEGMENT[class_name]
            segment_id = SEGMENT_TO_LABEL[class_id]

            las = laspy.read(str(las_file))
            coords = np.vstack((las.x, las.y, las.z)).T
            # -----------------------------
            # ✅ 步骤1：减去全局中心（对齐）
            # -----------------------------
            coords = coords - global_center
            # -----------------------------
            # ✅ 步骤2：局部中心化 + 单位球归一化
            # -----------------------------
            centroid = np.mean(coords, axis = 0)
            coords_centered = coords - centroid
            max_distance = np.max(np.linalg.norm(coords_centered, axis = 1))
            max_distance = max(max_distance, 1e-8)
            coords_normalized = coords_centered / max_distance
            coords_final = np.round(coords_normalized, 8)
            # -----------------------------
            # ✅ 步骤3：处理颜色
            # -----------------------------
            colors = np.vstack((las.red, las.green, las.blue)).T
            colors = (colors / 65535.0 * 255).astype(np.uint8)
            # -----------------------------
            # ✅ 步骤4：创建标签
            # -----------------------------
            segment20 = np.full(coords_final.shape[0], segment_id, dtype = np.uint8)
            normals = estimate_normals(coords_final)
            # -----------------------------
            # ✅ 步骤5：保存
            # -----------------------------
            all_coords.append(coords_final.astype(np.float32))
            all_colors.append(colors)
            all_segments.append(segment20)
            all_normals.append(normals.astype(np.float32))

        # -----------------------------
        # ✅ 步骤6：保存场景文件
        # -----------------------------
        # 合并所有文件的数据
        all_coords = np.concatenate(all_coords, axis = 0)
        all_colors = np.concatenate(all_colors, axis = 0)
        all_segments = np.concatenate(all_segments, axis = 0)
        all_normals = np.concatenate(all_normals, axis = 0)

        # 统计各类别点数量
        mask_0 = all_segments == 0
        mask_1 = all_segments == 1
        mask_2 = all_segments == 2
        mask_3 = all_segments == 3

        n0 = np.sum(mask_0)
        n1 = np.sum(mask_1)
        n2 = np.sum(mask_2)
        n3 = np.sum(mask_3)

        base_count = n0 + n3
        target_1 = int(base_count * ratio)
        target_2 = int(base_count * ratio)

        # 类别 1 采样
        indices_1 = np.where(mask_1)[0]
        if len(indices_1) > target_1:
            sampled_indices_1 = np.random.choice(indices_1, size = target_1, replace = False)
        else:
            sampled_indices_1 = indices_1
        mask_1_sparse = np.zeros_like(mask_1, dtype = bool)
        mask_1_sparse[sampled_indices_1] = True

        # 类别 2 采样
        indices_2 = np.where(mask_2)[0]
        if len(indices_2) > target_2:
            sampled_indices_2 = np.random.choice(indices_2, size = target_2, replace = False)
        else:
            sampled_indices_2 = indices_2
        mask_2_sparse = np.zeros_like(mask_2, dtype = bool)
        mask_2_sparse[sampled_indices_2] = True

        # 合并掩码并应用
        final_mask = mask_0 | mask_3 | mask_1_sparse | mask_2_sparse
        all_coords = all_coords[final_mask]
        all_colors = all_colors[final_mask]
        all_segments = all_segments[final_mask]
        all_normals = all_normals[final_mask]

        print(
            f"[{scene_name}] 类别点数: 0={n0}, 1原={n1}, 1保留={len(sampled_indices_1)}, 2原={n2}, 2保留={len(sampled_indices_2)}, 3={n3}, 总保留={len(all_coords)}")

        # 保存合并后的场景
        np.save(scene_dir / "coord.npy", all_coords)  # 坐标信息
        np.save(scene_dir / "color.npy", all_colors)  # 颜色信息
        np.save(scene_dir / "normal.npy", all_normals)  # 法向量信息
        np.save(scene_dir / "segment20.npy", all_segments)  # 语义标签信息


if __name__ == "__main__":
    ratio = 0.8  # 处理不平衡类别下采样倍率
    input_dir = r"D:\user\Documents\ai\三维重建\点云资料\输电人工智能\110kV牛莲线las点云--按类别提取"
    output_dir = "D:/user/code/AI/Point++/data/output"
    main(input_dir, output_dir, ratio)
