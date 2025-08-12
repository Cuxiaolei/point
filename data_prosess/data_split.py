import os
import random
import shutil
import numpy as np


def scale_point_cloud(coords, target_range=(-1, 1)):
    """将点云坐标缩放到指定范围"""
    if coords.size == 0:
        return coords

    # 计算每个维度的最小值和最大值
    min_val = np.min(coords, axis=0)
    max_val = np.max(coords, axis=0)

    # 处理极端情况（所有点坐标相同）
    range_val = max_val - min_val
    range_val[range_val < 1e-8] = 1e-8  # 避免除零

    # 线性缩放至目标范围
    scaled = (coords - min_val) / range_val
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled


def process_scene(scene_path, target_range=(-1, 1)):
    """处理单个场景，缩放点云坐标"""
    coord_path = os.path.join(scene_path, "coord.npy")
    if not os.path.exists(coord_path):
        print(f"  警告: 未找到坐标文件 {coord_path}，跳过处理")
        return

    # 加载并缩放坐标
    coords = np.load(coord_path)
    scaled_coords = scale_point_cloud(coords, target_range)

    # 保存缩放后的坐标（覆盖原文件）
    np.save(coord_path, scaled_coords)
    print(f"  已将坐标缩放到 {target_range} 范围")


def main():
    # ------------------------ 配置参数 ------------------------
    data_root = "D:/user/code/AI/Point++/data/output"  # 原始数据根目录
    train_ratio = 0.7  # 训练集比例
    output_root = "D:/user/code/AI/Point++/data/tower"  # 输出根目录
    train_dir_name = "train"
    val_dir_name = "val"
    target_range = (-1, 1)  # 点云缩放目标范围
    # --------------------------------------------------------

    # 1. 获取所有场景文件夹（不丢弃任何数据）
    scene_dirs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]
    print(f"找到 {len(scene_dirs)} 个场景文件夹（不丢弃任何数据）")

    if not scene_dirs:
        print("错误: 未找到任何场景文件夹")
        return

    # 2. 随机打乱场景列表（设置种子确保可复现）
    random.seed(42)
    random.shuffle(scene_dirs)

    # 3. 划分训练集和验证集
    split_index = int(len(scene_dirs) * train_ratio)
    train_scenes = scene_dirs[:split_index]
    val_scenes = scene_dirs[split_index:]

    print(f"划分结果: 训练集 {len(train_scenes)} 个，验证集 {len(val_scenes)} 个")

    # 4. 创建输出目录
    train_output = os.path.join(output_root, train_dir_name)
    val_output = os.path.join(output_root, val_dir_name)
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    # 5. 复制并处理训练集
    print("\n开始处理训练集...")
    for scene in train_scenes:
        src = os.path.join(data_root, scene)
        dst = os.path.join(train_output, scene)

        if os.path.exists(dst):
            print(f"  跳过已存在场景: {scene}")
            continue

        try:
            shutil.copytree(src, dst)
            print(f"  处理场景: {scene}")
            process_scene(dst, target_range)  # 缩放坐标
        except Exception as e:
            print(f"  处理失败 {scene}: {str(e)}")

    # 6. 复制并处理验证集
    print("\n开始处理验证集...")
    for scene in val_scenes:
        src = os.path.join(data_root, scene)
        dst = os.path.join(val_output, scene)

        if os.path.exists(dst):
            print(f"  跳过已存在场景: {scene}")
            continue

        try:
            shutil.copytree(src, dst)
            print(f"  处理场景: {scene}")
            process_scene(dst, target_range)  # 缩放坐标
        except Exception as e:
            print(f"  处理失败 {scene}: {str(e)}")

    print(f"\n处理完成！")
    print(f"训练集路径: {train_output}")
    print(f"验证集路径: {val_output}")
    print(f"点云坐标已缩放到 {target_range} 范围")


if __name__ == "__main__":
    main()