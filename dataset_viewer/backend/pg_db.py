"""
PostgreSQL 异步读库：当 DATABASE_URL 设置时使用 asyncpg 提供 runs/episodes/images 查询。
与 db.py 的 SQLite 语义一致，供 main.py 统一「查库」。
"""
import os
from typing import Any

import asyncpg

DEFAULT_DATABASE_URL = "postgresql://admin:password123@localhost:5432/annotation_system"

_pool: asyncpg.Pool | None = None


def get_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return url if url and url.strip() else None


async def get_pool() -> asyncpg.Pool | None:
    global _pool
    if _pool is not None:
        return _pool
    url = get_database_url()
    if not url:
        return None
    try:
        _pool = await asyncpg.create_pool(url, min_size=1, max_size=5, command_timeout=60)
        return _pool
    except Exception:
        return None


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    return dict(row)


# ---------- 读接口（与 db.py 语义一致） ----------


async def get_db_stats() -> dict:
    """库内统计：runs, episodes, images。用于部署校验。"""
    pool = await get_pool()
    if not pool:
        return {"runs": 0, "episodes": 0, "images": 0}
    async with pool.acquire() as conn:
        runs = await conn.fetchval("SELECT COUNT(*) FROM runs")
        episodes = await conn.fetchval("SELECT COUNT(*) FROM episodes")
        images = await conn.fetchval("SELECT COUNT(*) FROM images")
        return {"runs": runs or 0, "episodes": episodes or 0, "images": images or 0}


async def get_runs() -> list[str]:
    """返回 run 名称列表"""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM runs ORDER BY name")
        return [r["name"] for r in rows]


async def get_episodes(run_name: str) -> list[dict[str, Any]]:
    """返回该 run 下各 episode 摘要：name, imageCount, annotatedCount"""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        run_id = await conn.fetchval("SELECT id FROM runs WHERE name = $1", run_name)
        if run_id is None:
            return []
        rows = await conn.fetch(
            "SELECT name, image_count, annotated_count FROM episodes WHERE run_id = $1 ORDER BY name",
            run_id,
        )
        return [
            {"name": r["name"], "imageCount": r["image_count"], "annotatedCount": r["annotated_count"]}
            for r in rows
        ]


async def get_images_cursor(
    run_name: str, ep_name: str, cursor: str | None, limit: int = 50
) -> tuple[list[dict[str, Any]], str | None]:
    """游标分页：返回 (items, next_cursor)。items 为 { id, filename, annotated }"""
    pool = await get_pool()
    if not pool:
        return [], None
    async with pool.acquire() as conn:
        run_id = await conn.fetchval("SELECT id FROM runs WHERE name = $1", run_name)
        if run_id is None:
            return [], None
        ep_id = await conn.fetchval(
            "SELECT id FROM episodes WHERE run_id = $1 AND name = $2", run_id, ep_name
        )
        if ep_id is None:
            return [], None
        if cursor is None:
            rows = await conn.fetch(
                """SELECT path_id, filename, annotated FROM images
                   WHERE episode_id = $1 ORDER BY sort_key ASC LIMIT $2""",
                ep_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                """SELECT path_id, filename, annotated FROM images
                   WHERE episode_id = $1 AND sort_key > $2 ORDER BY sort_key ASC LIMIT $3""",
                ep_id,
                cursor,
                limit + 1,
            )
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [
            {"id": r["path_id"], "filename": r["filename"], "annotated": bool(r["annotated"])}
            for r in rows
        ]
        next_cursor = rows[-1]["filename"] if has_more and rows else None
        return items, next_cursor


async def get_image_by_path_id(path_id: str) -> dict | None:
    """根据 path_id 查询 images 表，返回 { id, path_id } 或 None"""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, path_id FROM images WHERE path_id = $1", path_id)
        return dict(row) if row else None


async def get_annotation_boxes(image_id: int) -> list:
    """查询 annotations 表的 boxes"""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT boxes FROM annotations WHERE image_id = $1", image_id)
        if not row or not row["boxes"]:
            return []
        return row["boxes"] if isinstance(row["boxes"], list) else []


async def upsert_annotation(image_id: int, boxes: list) -> None:
    """UPSERT annotations"""
    import json
    pool = await get_pool()
    if not pool:
        return
    boxes_json = json.dumps(boxes) if not isinstance(boxes, str) else boxes
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO annotations (image_id, boxes, updated_at)
               VALUES ($1, $2, NOW())
               ON CONFLICT (image_id) DO UPDATE SET boxes = $2, updated_at = NOW()""",
            image_id,
            boxes_json,
        )


async def update_image_annotated(image_id: int, annotated: bool = True) -> None:
    """更新 images.annotated 并刷新对应 episode 的 annotated_count"""
    pool = await get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE images SET annotated = $1 WHERE id = $2",
            1 if annotated else 0,
            image_id,
        )
        await conn.execute(
            """UPDATE episodes SET annotated_count = (
                SELECT COUNT(*)::int FROM images WHERE episode_id = episodes.id AND annotated = 1
            ) WHERE id = (SELECT episode_id FROM images WHERE id = $1)""",
            image_id,
        )


async def get_all_annotated_for_export() -> list[dict[str, Any]]:
    """获取所有已标注图片：run, ep, path_id, filename, boxes。用于导出"""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT r.name AS run_name, e.name AS ep_name, i.path_id, i.filename, a.boxes
               FROM annotations a
               JOIN images i ON a.image_id = i.id
               JOIN episodes e ON i.episode_id = e.id
               JOIN runs r ON e.run_id = r.id
               ORDER BY r.name, e.name, i.sort_key"""
        )
        return [
            {
                "run": r["run_name"],
                "ep": r["ep_name"],
                "path_id": r["path_id"],
                "filename": r["filename"],
                "boxes": r["boxes"] if isinstance(r["boxes"], list) else [],
            }
            for r in rows
        ]


async def get_images_page(page: int = 1, page_size: int = 50) -> list[dict[str, Any]]:
    """扁平分页：按 id 排序，返回 id, file_path, status。用于 GET /api/images"""
    pool = await get_pool()
    if not pool:
        return []
    offset = (page - 1) * page_size
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, path_id, annotated FROM images
               ORDER BY id ASC LIMIT $1 OFFSET $2""",
            page_size,
            offset,
        )
        return [
            {
                "id": r["path_id"],
                "file_path": r["path_id"],
                "status": "annotated" if r["annotated"] else "pending",
            }
            for r in rows
        ]
