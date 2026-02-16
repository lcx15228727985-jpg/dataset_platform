#!/usr/bin/env python3
"""
统一转换脚本：根据目录结构将各 ep 的 image(s) 文件夹内非 PNG 图片转为 PNG。

目录约定（按优先级尝试）：
  1) dataset/<run>/episode_0/images, episode_1/images, ... episode_12/images
  2) dataset/<run>/episode_0/image,  episode_1/image,  ... episode_12/image
  3) dataset/ep0/images, ep1/images, ... ep12/images
  4) dataset/ep0/image,  ep1/image,  ... ep12/image

用法：
  python dataset_viewer/convert_images_to_png.py [数据根目录]
  不传参数时默认使用：脚本所在目录的上一级/dataset
"""
from __future__ import print_function

import os
import sys

# 支持的图片后缀（转为 PNG 时只处理非 .png）
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}


def get_default_root():
    script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    return os.path.normpath(os.path.join(script_dir, "..", "dataset"))


def find_image_folders(root):
    """
    根据目录结构找出所有需要转换的 images 文件夹。
    返回 [(显示名, 绝对路径), ...]
    """
    if not os.path.isdir(root):
        return []
    root = os.path.abspath(root)
    found = []

    # 1) 先看 root 下是否有 run 子目录（如 dataset0209）
    for run_name in sorted(os.listdir(root)):
        run_path = os.path.join(root, run_name)
        if not os.path.isdir(run_path):
            continue
        # episode_0 .. episode_12 下 images 或 image
        for i in range(13):
            for sub in ("images", "image"):
                ep_dir = os.path.join(run_path, "episode_{}".format(i))
                img_dir = os.path.join(ep_dir, sub)
                if os.path.isdir(img_dir):
                    found.append(("{}/episode_{}/{}".format(run_name, i, sub), img_dir))
                    break
        if found and run_name in found[-1][0]:
            continue
        # 若没有 episode_*，再试 ep0..ep12
        for i in range(13):
            for sub in ("images", "image"):
                ep_dir = os.path.join(run_path, "ep{}".format(i))
                img_dir = os.path.join(ep_dir, sub)
                if os.path.isdir(img_dir):
                    found.append(("{}/ep{}/{}".format(run_name, i, sub), img_dir))
                    break

    # 2) 若 root 下直接是 ep0..ep12
    if not found:
        for i in range(13):
            for sub in ("images", "image"):
                ep_dir = os.path.join(root, "ep{}".format(i))
                img_dir = os.path.join(ep_dir, sub)
                if os.path.isdir(img_dir):
                    found.append(("ep{}/{}".format(i, sub), img_dir))
                    break

    return found


def convert_one_folder(folder_path):
    """
    将 folder_path 内所有非 .png 的图片转为同主名 .png 并删除原文件。
    返回 (转换数量, 错误列表)
    """
    try:
        from PIL import Image
    except ImportError:
        return 0, ["需要安装 Pillow: pip install Pillow"]
    if not os.path.isdir(folder_path):
        return 0, []
    converted = 0
    errors = []
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMAGE_EXT or ext == ".png":
            continue
        stem = os.path.splitext(f)[0]
        png_path = os.path.join(folder_path, stem + ".png")
        try:
            img = Image.open(path).convert("RGB")
            img.save(png_path, "PNG")
            os.remove(path)
            converted += 1
            print("  转: {} -> {}.png".format(f, stem))
        except Exception as e:
            err_msg = "{}: {}".format(f, e)
            errors.append(err_msg)
            print("  失败: {}".format(err_msg), file=sys.stderr)
    return converted, errors


def main():
    root = os.environ.get("DATA_ROOT") or (sys.argv[1] if len(sys.argv) > 1 else get_default_root())
    root = os.path.normpath(os.path.abspath(root))
    print("数据根目录: {}".format(root))
    if not os.path.isdir(root):
        print("错误: 目录不存在。", file=sys.stderr)
        sys.exit(1)

    folders = find_image_folders(root)
    if not folders:
        print("未找到符合约定的 image(s) 目录（episode_0..12/images 或 ep0..12/images）。")
        sys.exit(0)

    print("找到 {} 个 image 目录，开始转换...".format(len(folders)))
    total = 0
    all_errors = []
    for name, path in folders:
        print("\n[{}] {}".format(name, path))
        n, errs = convert_one_folder(path)
        total += n
        if errs:
            all_errors.extend(["{}: {}".format(name, e) for e in errs])
        if n == 0 and not errs:
            print("  (无非 PNG 文件)")
    print("\n----------")
    print("转换完成，共 {} 张转为 PNG。".format(total))
    if all_errors:
        print("失败 {} 个:".format(len(all_errors)), file=sys.stderr)
        for e in all_errors:
            print("  ", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
