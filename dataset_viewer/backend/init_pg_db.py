"""
PostgreSQL 数据初始化：连接 Docker 中的 Postgres，创建 runs/episodes/images 表，
并扫描 DATA_ROOT 下所有 run/episode 的图片元数据写入数据库。
运行: python -m backend.init_pg_db  （在 dataset_viewer 目录下）
环境变量:
  DATA_ROOT   - 数据根目录，默认 ../dataset（相对 backend 的上级）
  DATABASE_URL - 连接串，默认 postgresql://admin:password123@localhost:5432/annotation_system
"""
import asyncio
import os
import sys

# 确保在 dataset_viewer 下运行，以便 backend 包可导入
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
os.chdir(_app_dir)

import asyncpg

from backend.data import (
    get_data_root,
    get_run_folders,
    list_ep_folders,
    IMAGE_EXT,
    get_annotation_path,
)

DEFAULT_DATABASE_URL = "postgresql://admin:password123@localhost:5432/annotation_system"

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0,
    annotated_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, name)
);

CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    path_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    annotated INTEGER NOT NULL DEFAULT 0,
    sort_key TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_episode_sort ON images(episode_id, sort_key);
CREATE INDEX IF NOT EXISTS idx_images_annotated ON images(episode_id, annotated);
CREATE UNIQUE INDEX IF NOT EXISTS idx_images_path_id ON images(path_id);

CREATE TABLE IF NOT EXISTS annotations (
    image_id INTEGER PRIMARY KEY REFERENCES images(id),
    boxes JSONB DEFAULT '[]',
    is_reviewed BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW()
);
"""


async def init_db(data_root: str, conn: asyncpg.Connection) -> dict:
    """建表并扫描 data_root 写入 runs/episodes/images。返回统计."""
    # 1. 建表
    for stmt in PG_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            await conn.execute(stmt)

    # 2. 清空旧数据（按外键顺序）
    await conn.execute("DELETE FROM annotations")
    await conn.execute("DELETE FROM images")
    await conn.execute("DELETE FROM episodes")
    await conn.execute("DELETE FROM runs")

    total_images = 0
    runs = get_run_folders(data_root)

    for run_name in runs:
        await conn.execute("INSERT INTO runs (name) VALUES ($1) ON CONFLICT (name) DO NOTHING", run_name)
        row = await conn.fetchrow("SELECT id FROM runs WHERE name = $1", run_name)
        run_id = row["id"]

        ep_list = list_ep_folders(data_root, run_name)
        for ep_name, image_folder in ep_list:
            if not image_folder or not os.path.isdir(image_folder):
                await conn.execute(
                    "INSERT INTO episodes (run_id, name, image_count, annotated_count) VALUES ($1, $2, 0, 0)",
                    run_id, ep_name,
                )
                continue

            paths = []
            for f in os.listdir(image_folder):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXT:
                    paths.append(os.path.join(image_folder, f))
            paths.sort(key=lambda p: os.path.basename(p))
            annotated_count = sum(1 for p in paths if os.path.isfile(get_annotation_path(p)))

            await conn.execute(
                "INSERT INTO episodes (run_id, name, image_count, annotated_count) VALUES ($1, $2, $3, $4)",
                run_id, ep_name, len(paths), annotated_count,
            )
            ep_id = await conn.fetchval("SELECT id FROM episodes WHERE run_id = $1 AND name = $2", run_id, ep_name)

            # 批量插入当前 episode 的 images
            data = []
            for p in paths:
                ann = 1 if os.path.isfile(get_annotation_path(p)) else 0
                rel = os.path.relpath(p, data_root).replace("\\", "/")
                filename = os.path.basename(p)
                sort_key = filename
                data.append((ep_id, rel, filename, ann, sort_key))
            if data:
                await conn.executemany(
                    "INSERT INTO images (episode_id, path_id, filename, annotated, sort_key) VALUES ($1, $2, $3, $4, $5)",
                    data,
                )
            total_images += len(paths)

    return {"runs": len(runs), "total_images": total_images}


async def main():
    data_root = get_data_root()
    if not os.path.isdir(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    print(f"数据根目录: {data_root}")
    print(f"连接数据库: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    print("正在建表并扫描录入（大目录可能需要数分钟）...")

    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        print(f"连接失败: {e}")
        sys.exit(1)

    try:
        stats = await init_db(data_root, conn)
        print(f"完成. runs={stats['runs']}, total_images={stats['total_images']}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
