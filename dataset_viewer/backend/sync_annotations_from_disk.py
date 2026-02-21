"""
从磁盘 *_annot.json 同步到 catalog.db 的 annotations 表。
用于恢复备份后或 scan_dataset 后，将主标注页保存的标注写入 DB，便于导出与工作台展示。

运行: python -m backend.sync_annotations_from_disk  （在 dataset_viewer 目录下）
"""
import os
import sys
from pathlib import Path


def main():
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    os.chdir(app_dir)

    from dotenv import load_dotenv
    load_dotenv(Path(app_dir) / ".env")

    from backend.data import get_data_root, get_run_folders, list_ep_folders, get_annotation_path, load_annotation
    from backend.db import get_db_path, get_image_by_path_id, upsert_annotation, update_image_annotated

    data_root = get_data_root()
    if not os.path.isdir(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        sys.exit(1)
    db_path = get_db_path()
    if not os.path.isfile(db_path):
        print(f"错误: catalog.db 不存在，请先运行 python -m backend.scan_dataset")
        sys.exit(1)

    synced = 0
    skipped = 0
    for run_name in get_run_folders(data_root):
        for ep_name, image_folder in list_ep_folders(data_root, run_name):
            if not image_folder or not os.path.isdir(image_folder):
                continue
            for f in os.listdir(image_folder):
                ext = os.path.splitext(f)[1].lower()
                if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}:
                    continue
                img_path = os.path.join(image_folder, f)
                ann_path = get_annotation_path(img_path)
                if not os.path.isfile(ann_path):
                    continue
                rel = os.path.relpath(img_path, data_root).replace("\\", "/")
                img_row = get_image_by_path_id(db_path, rel)
                if not img_row:
                    skipped += 1
                    continue
                ann = load_annotation(img_path)
                objs = ann.get("objects") or ann.get("boxes") or []
                boxes = objs if isinstance(objs, list) else []
                upsert_annotation(db_path, img_row["id"], boxes, None)
                update_image_annotated(db_path, img_row["id"], annotated=len(boxes) > 0)
                synced += 1

    print(f"完成. 同步 {synced} 条标注到 DB，跳过 {skipped} 条（不在 catalog 中）")


if __name__ == "__main__":
    main()
