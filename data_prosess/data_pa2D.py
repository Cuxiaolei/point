#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click LAS -> 2DPASS(SemanticKITTI-style) converter.

- 输入: --in 指向含 .las 的根目录（递归）
- 输出: --out/sequences/<seq>/{velodyne,labels}/xxxxxx.{bin,label}
- .bin: float32 [x,y,z,intensity]
- .label: uint32 (instance<<16 | semantic), 此处 instance 恒为 0
- 区域到 sequence: 取文件名前缀 "A-B" (如 1-2(1_2)_变电站.las => "1-2")，为每个唯一键分配 00,01,...
- 类别映射(中文关键字):
    铁塔类(0):   ["铁塔","绝缘子"]
    地面类(1):   ["建筑物","公路","低点","地面点","变电站","中等植被点"]
    输电线(2):   ["导线","引流线","地线"]
- 标注优先级:
    1) 文件名末尾中文类关键词命中 -> 整文件按该类
    2) 否则用点级 classification 数值做粗映射(见 numeric_fallback_map)
       映射不到的记 255(ignore)，并汇总打印
"""
import os, re, argparse
from pathlib import Path
import numpy as np
import laspy
from tqdm import tqdm
from collections import defaultdict

IGNORE = 255

# 关键词到 3 类 id
KEYWORDS = {
    0: ["铁塔","绝缘子"],
    1: ["建筑物","公路","低点","地面点","变电站","中等植被点"],
    2: ["导线","引流线","地线"],
}
# classification 数值的兜底映射（常见 LAS 编码：2地面,3低植被,4中植被,5高植被,6建筑物）
# 注意：电力场景自定义编号无法穷举，这里仅把“地面/建筑/植被”并到 1 类，其他未知为 255
numeric_fallback_map = {
    2: 1,  # Ground
    3: 1,  # Low vegetation
    4: 1,  # Medium vegetation
    5: 1,  # High vegetation
    6: 1,  # Building
}

REGION_RE = re.compile(r"^([0-9]+-[0-9]+)")
CLASS_TAIL_RE = re.compile(r"(铁塔|绝缘子|建筑物|公路|低点|地面点|变电站|中等植被点|导线|引流线|地线)(?:\.las)$")

def get_region_key(stem:str)->str:
    m = REGION_RE.match(stem)
    return m.group(1) if m else "00-00"

def guess_class_from_name(stem:str):
    m = CLASS_TAIL_RE.search(stem)
    if not m: return None
    kw = m.group(1)
    for gid, kws in KEYWORDS.items():
        if any(kw == k for k in kws):
            return gid
    return None

def normalize_intensity(arr: np.ndarray)->np.ndarray:
    # las.intensity 通常为 uint16；归一化到 [0,1]
    if np.issubdtype(arr.dtype, np.integer):
        maxv = 65535.0 if arr.dtype.itemsize >= 2 else 255.0
        return (arr.astype(np.float32) / maxv).clip(0.0, 1.0)
    vmax = float(arr.max()) if arr.size else 1.0
    return (arr.astype(np.float32) / (vmax if vmax>0 else 1.0)).clip(0.0,1.0)

def encode_label_uint32(sem: np.ndarray)->np.ndarray:
    return (sem.astype(np.uint32) & np.uint32(0xFFFF))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_root", required=True, help="输入 .las 根目录（递归）")
    ap.add_argument("--out", dest="out_root", required=True, help="输出根目录（自动创建 sequences/*）")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="按区域划分验证集比例（写出 splits）")
    ap.add_argument("--test_ratio", type=float, default=0.0, help="按区域划分测试集比例（写出 splits）")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    las_paths = sorted(p for p in in_root.rglob("*.las"))
    if not las_paths:
        raise SystemExit(f"在 {in_root} 下未找到 .las")

    # 1) 建立区域 -> sequence id
    region_keys = [get_region_key(p.stem) for p in las_paths]
    uniq_regions = sorted(set(region_keys))
    region2seq = {rk: f"{i:02d}" for i, rk in enumerate(uniq_regions)}
    # 创建目录
    for sid in region2seq.values():
        (out_root/"sequences"/sid/"velodyne").mkdir(parents=True, exist_ok=True)
        (out_root/"sequences"/sid/"labels").mkdir(parents=True, exist_ok=True)

    # 2) 划分 train/val/test（写一个 splits 文件，训练时可参考）
    n = len(uniq_regions)
    n_test = int(round(n*args.test_ratio))
    n_val  = int(round(n*args.val_ratio))
    test_regions = set(uniq_regions[:n_test])
    val_regions  = set(uniq_regions[n_test:n_test+n_val])
    train_regions= set(uniq_regions[n_test+n_val:])
    splits = {
        "train": [region2seq[r] for r in sorted(train_regions)],
        "val":   [region2seq[r] for r in sorted(val_regions)],
        "test":  [region2seq[r] for r in sorted(test_regions)],
    }
    (out_root/"custom_splits.txt").write_text(
        "train: "+",".join(splits["train"])+"\n"+
        "val: "  +",".join(splits["val"])  +"\n"+
        "test: " +",".join(splits["test"]) +"\n",
        encoding="utf-8"
    )

    # 3) 转换
    per_seq_counter = defaultdict(int)
    unknown_numeric_codes = set()
    name_hit, numeric_hit, ignored_cnt = 0,0,0

    for p, rk in tqdm(list(zip(las_paths, region_keys)), desc="Converting"):
        seq = region2seq[rk]
        idx = per_seq_counter[seq]; per_seq_counter[seq]+=1
        scan = f"{idx:06d}"
        out_bin = out_root/"sequences"/seq/"velodyne"/f"{scan}.bin"
        out_lab = out_root/"sequences"/seq/"labels"/f"{scan}.label"

        las = laspy.read(p)
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        intensity = normalize_intensity(np.asarray(las.intensity))
        pts = np.column_stack([xyz, intensity])

        # 先看文件名能否给出 11 类中的某一类（整文件一致）
        gid = guess_class_from_name(p.stem)
        if gid is not None:
            sem = np.full((pts.shape[0],), gid, dtype=np.int32)
            name_hit += 1
        else:
            # 否则：按 classification 数值兜底映射
            raw = np.asarray(las.classification).astype(np.int32) if hasattr(las, "classification") else None
            if raw is None or raw.size == 0:
                sem = np.full((pts.shape[0],), IGNORE, dtype=np.int32)
                ignored_cnt += sem.size
            else:
                sem = np.full_like(raw, IGNORE)
                # 把常见“地面/建筑/植被”并到 1 类
                for k,v in numeric_fallback_map.items():
                    sem[raw==k] = v
                # 统计未知编码
                unk = np.unique(raw[(sem==IGNORE)])
                unknown_numeric_codes.update(map(int, unk.tolist()))
                ignored_cnt += int((sem==IGNORE).sum())
                numeric_hit += 1

        # 写出
        pts.astype(np.float32).tofile(out_bin)
        encode_label_uint32(sem).tofile(out_lab)

    # 4) 写一个最小 label_mapping（供 2DPASS 配置引用）
    lm = (
        "labels: {0: 铁塔类, 1: 地面类, 2: 输电线类}\n"
        "learning_map: {0: 0, 1: 1, 2: 2, 255: 255}\n"
        "learning_map_inv: {0: 0, 1: 1, 2: 2}\n"
        "learning_ignore: {255: true}\n"
        "color_map: {0: [255,0,0], 1: [0,255,0], 2: [0,0,255]}\n"
    )
    (out_root/"your-3cls-label-mapping.yaml").write_text(lm, encoding="utf-8")

    # 5) 日志总结
    print("\n=== Summary ===")
    print(f"Total files: {len(las_paths)}")
    print(f"Name-based class hits: {name_hit} file(s)")
    print(f"Numeric(classification) fallback used on: {numeric_hit} file(s)")
    if unknown_numeric_codes:
        print("Unknown classification codes mapped to IGNORE (255):", sorted(unknown_numeric_codes))
        print("（如需更精确，请把这些编号并入脚本顶部的 numeric_fallback_map）")
    print(f"Total ignored points (label=255): {ignored_cnt}")
    print("Output root:", str(out_root.resolve()))
    print("Sequences split written to:", str((out_root/'custom_splits.txt').resolve()))
    print("Label mapping yaml:", str((out_root/'your-3cls-label-mapping.yaml').resolve()))
    print("================\n")
    print("提示：如果你没有相机图像，训练 2DPASS 时建议 baseline-only：")
    print("python main.py --log_dir run3cls --config config/2DPASS-semantickitti.yaml --gpu 0 --baseline_only")
    print("把 num_classes 改成 3，并把 label_mapping 指到上面的 your-3cls-label-mapping.yaml")

if __name__ == "__main__":
    main()
