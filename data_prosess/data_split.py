import os
import random
import shutil
import numpy as np
from pathlib import Path
import json

def scale_point_cloud(coords, target_range=(-1, 1)):
    if coords.size == 0:
        return coords
    min_val = np.min(coords, axis=0)
    max_val = np.max(coords, axis=0)
    range_val = max_val - min_val
    range_val[range_val < 1e-8] = 1e-8
    scaled = (coords - min_val) / range_val
    scaled = scaled * (target_range[1] - target_range[0]) + target_range[0]
    return scaled

def main():
    data_root = Path("D:/user/code/AI/Point++/data/output")
    output_root = Path("D:/user/code/AI/Point++/data/tower")
    train_ratio = 0.7
    target_range = (-1, 1)

    scene_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"找到 {len(scene_dirs)} 个场景")

    random.seed(42)
    random.shuffle(scene_dirs)

    split_index = int(len(scene_dirs) * train_ratio)
    train_scenes = scene_dirs[:split_index]
    val_scenes = scene_dirs[split_index:]

    # 保存划分信息
    split_info = {
        "train": [s.name for s in train_scenes],
        "val": [s.name for s in val_scenes]
    }
    with open(output_root / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"训练集 {len(train_scenes)} 个，验证集 {len(val_scenes)} 个")

    for subset_name, subset_scenes in [("train", train_scenes), ("val", val_scenes)]:
        subset_path = output_root / subset_name
        subset_path.mkdir(parents=True, exist_ok=True)

        for scene in subset_scenes:
            dst = subset_path / scene.name
            dst.mkdir(exist_ok=True)
            # 复制并归一化坐标
            coords = np.load(scene / "coord.npy")
            coords_scaled = scale_point_cloud(coords, target_range)
            np.save(dst / "coord.npy", coords_scaled)

            # 其他文件直接复制
            for fname in ["color.npy", "normal.npy", "label.npy"]:
                shutil.copy(scene / fname, dst / fname)

    print("数据划分和归一化完成！")

if __name__ == "__main__":
    main()
