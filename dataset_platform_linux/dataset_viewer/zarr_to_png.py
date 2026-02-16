#!/usr/bin/env python3
"""
将 Zarr 格式的 images（如 dataset0209/episode_0/images 下的 0.0.0.0 等 chunk）
导出为可查看的 PNG 文件。

目录约定：
  dataset/<run>/episode_0/images/  <- Zarr 数组（.zarray + 0.0.0.0 等）
  dataset/<run>/episode_0/images_png/  <- 导出后的 frame_00000.png, frame_00001.png, ...

用法：
  python dataset_viewer/zarr_to_png.py [数据根目录]
  不传参数时默认：脚本上一级/dataset
"""
from __future__ import print_function

import os
import sys

import numpy as np
try:
    import zarr
except ImportError:
    print("请先安装 zarr: pip install zarr", file=sys.stderr)
    sys.exit(1)
try:
    from PIL import Image
except ImportError:
    print("请先安装 Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(1)


def get_default_root():
    script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    return os.path.normpath(os.path.join(script_dir, "..", "dataset"))


def find_zarr_image_folders(root):
    """
    找出 root 下所有作为 Zarr 数组的 images 目录（存在 .zarray）。
    返回 [(显示名, images_dir 绝对路径), ...]
    """
    if not os.path.isdir(root):
        return []
    root = os.path.abspath(root)
    found = []
    for run_name in sorted(os.listdir(root)):
        run_path = os.path.join(root, run_name)
        if not os.path.isdir(run_path):
            continue
        for i in range(13):
            images_dir = os.path.join(run_path, "episode_{}".format(i), "images")
            if not os.path.isdir(images_dir):
                continue
            zarray_path = os.path.join(images_dir, ".zarray")
            if os.path.isfile(zarray_path):
                found.append(("{}/episode_{}/images".format(run_name, i), images_dir))
    return found


def export_zarr_to_png(images_dir, out_subdir="images_png"):
    """
    将 Zarr 数组 images_dir 的每一帧导出为 PNG，保存到同级的 out_subdir 下。
    例如 episode_0/images -> episode_0/images_png/frame_00000.png ...
    返回 (导出张数, 错误信息)
    """
    if not os.path.isdir(images_dir):
        return 0, "目录不存在"
    zarray_path = os.path.join(images_dir, ".zarray")
    if not os.path.isfile(zarray_path):
        return 0, "非 Zarr 数组（无 .zarray）"
    try:
        arr = zarr.open(images_dir, mode="r")
    except Exception as e:
        return 0, "zarr.open 失败: {}".format(e)
    if arr.ndim < 2:
        return 0, "数组维度不足"
    # 约定：第一维为帧数，后两维为高宽，最后一维可能为通道
    n_frames = arr.shape[0]
    parent = os.path.dirname(images_dir)
    out_dir = os.path.join(parent, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    n_width = len(str(n_frames))
    fmt = "frame_{{:0{}d}}.png".format(n_width)
    exported = 0
    for i in range(n_frames):
        try:
            frame = arr[i]
            if hasattr(frame, "compute"):
                frame = frame.compute()
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                img = Image.fromarray(frame)
            elif frame.ndim == 2:
                img = Image.fromarray(frame)
            else:
                # 多通道或单通道
                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame.squeeze(-1)
                img = Image.fromarray(frame)
            out_path = os.path.join(out_dir, fmt.format(i))
            img.save(out_path)
            exported += 1
            if (i + 1) % 500 == 0 or i == 0:
                print("  已导出 {} / {} 帧".format(i + 1, n_frames))
        except Exception as e:
            print("  帧 {} 失败: {}".format(i, e), file=sys.stderr)
    return exported, None


def main():
    root = os.environ.get("DATA_ROOT") or (sys.argv[1] if len(sys.argv) > 1 else get_default_root())
    root = os.path.normpath(os.path.abspath(root))
    print("数据根目录: {}".format(root))
    if not os.path.isdir(root):
        print("错误: 目录不存在。", file=sys.stderr)
        sys.exit(1)

    folders = find_zarr_image_folders(root)
    if not folders:
        print("未找到 Zarr 格式的 episode_*/images 目录。")
        sys.exit(0)

    print("找到 {} 个 Zarr images 目录，开始导出 PNG...\n".format(len(folders)))
    total = 0
    for name, images_dir in folders:
        print("[{}]".format(name))
        n, err = export_zarr_to_png(images_dir)
        if err:
            print("  跳过: {}".format(err), file=sys.stderr)
        else:
            print("  导出 {} 张 -> 同级 images_png/".format(n))
            total += n
    print("\n----------")
    print("导出完成，共 {} 张 PNG。".format(total))
    print("可视化平台可改为使用各 episode 下的 images_png 目录查看。")
    sys.exit(0)


if __name__ == "__main__":
    main()
