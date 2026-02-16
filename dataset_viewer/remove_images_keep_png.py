#!/usr/bin/env python3
"""
删除各 episode 下的 images 目录（含 Zarr chunk 等），仅保留 images_png，以减小项目体积。

约定：仅当 episode_N/images_png 存在且可读时才删除 episode_N/images。
用法：
  python dataset_viewer/remove_images_keep_png.py [数据根目录] [--dry-run]
  不传参数时默认：DATA_ROOT 或 脚本上一级/dataset
  --dry-run：只打印将要删除的路径，不实际删除
"""
from __future__ import print_function

import os
import sys
import shutil


def get_default_root():
    script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    return os.path.normpath(os.path.join(script_dir, "..", "dataset"))


def main():
    root = os.environ.get("DATA_ROOT") or get_default_root()
    dry_run = "--dry-run" in sys.argv
    if "--dry-run" in sys.argv:
        sys.argv.remove("--dry-run")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        root = os.path.abspath(sys.argv[1])

    if not os.path.isdir(root):
        print("数据根目录不存在:", root, file=sys.stderr)
        sys.exit(1)

    removed = []
    for run_name in sorted(os.listdir(root)):
        run_path = os.path.join(root, run_name)
        if not os.path.isdir(run_path):
            continue
        for i in range(13):
            episode_dir = os.path.join(run_path, "episode_{}".format(i))
            images_dir = os.path.join(episode_dir, "images")
            images_png_dir = os.path.join(episode_dir, "images_png")
            if not os.path.isdir(images_dir):
                continue
            if not os.path.isdir(images_png_dir):
                print("跳过 {}（无 images_png）".format(images_dir))
                continue
            if dry_run:
                print("[dry-run] 将删除:", images_dir)
                removed.append(images_dir)
                continue
            try:
                shutil.rmtree(images_dir)
                print("已删除:", images_dir)
                removed.append(images_dir)
            except Exception as e:
                print("删除失败 {}: {}".format(images_dir, e), file=sys.stderr)

    if dry_run:
        print("--dry-run: 共 {} 个 images 目录将被删除。去掉 --dry-run 后执行以实际删除。".format(len(removed)))
    else:
        print("共删除 {} 个 images 目录，已保留对应 images_png。".format(len(removed)))


if __name__ == "__main__":
    main()
