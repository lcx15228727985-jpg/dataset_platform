"""
一次性扫描脚本：将 DATA_ROOT 下所有 run/episode 的图片元数据写入 SQLite。
运行: python -m backend.scan_dataset  （在 dataset_viewer 目录下）
环境变量 DATA_ROOT 可覆盖数据根目录。
"""
import os
import sys

def main():
    # 确保在 dataset_viewer 下运行，以便 backend 包可导入
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    os.chdir(app_dir)

    from backend.data import get_data_root
    from backend.db import get_db_path, scan_and_fill

    data_root = get_data_root()
    if not os.path.isdir(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        sys.exit(1)
    db_path = get_db_path()
    print(f"数据根目录: {data_root}")
    print(f"数据库路径: {db_path}")
    print("开始扫描并写入元数据（大目录可能需要数分钟）...")
    stats = scan_and_fill(db_path, data_root)
    print(f"完成. runs={stats['runs']}, total_images={stats['total_images']}")


if __name__ == "__main__":
    main()
