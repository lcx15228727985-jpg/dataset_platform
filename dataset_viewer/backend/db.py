"""
SQLite 元数据层：一次扫描入库，后续查询走索引，支持游标分页。
用于 45k+ 体量时替代每次 os.listdir。
"""
import os
import sqlite3
from pathlib import Path

from .data import (
    get_data_root,
    get_run_folders,
    list_ep_folders,
    IMAGE_EXT,
    get_annotation_path,
    load_annotation,
)

DB_NAME = "catalog.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0,
    annotated_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL,
    path_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    annotated INTEGER NOT NULL DEFAULT 0,
    sort_key TEXT NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_images_episode_sort ON images(episode_id, sort_key);
CREATE INDEX IF NOT EXISTS idx_images_annotated ON images(episode_id, annotated);
CREATE UNIQUE INDEX IF NOT EXISTS idx_images_path_id ON images(path_id);

CREATE TABLE IF NOT EXISTS annotations (
    image_id INTEGER PRIMARY KEY REFERENCES images(id),
    boxes TEXT DEFAULT '[]',
    is_reviewed INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


def get_db_path():
    root = get_data_root()
    return os.path.join(root, DB_NAME)


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def scan_and_fill(db_path: str, data_root: str) -> dict:
    """扫描 data_root 下所有 run/episode 图片，写入数据库。返回统计."""
    conn = init_db(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM annotations")
    cur.execute("DELETE FROM images")
    cur.execute("DELETE FROM episodes")
    cur.execute("DELETE FROM runs")
    conn.commit()

    total_images = 0
    runs = get_run_folders(data_root)
    for run_name in runs:
        cur.execute("INSERT OR IGNORE INTO runs (name) VALUES (?)", (run_name,))
        run_id = cur.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()[0]
        ep_list = list_ep_folders(data_root, run_name)
        for ep_name, image_folder in ep_list:
            if not image_folder or not os.path.isdir(image_folder):
                cur.execute(
                    "INSERT INTO episodes (run_id, name, image_count, annotated_count) VALUES (?, ?, 0, 0)",
                    (run_id, ep_name),
                )
                continue
            paths = []
            for f in os.listdir(image_folder):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXT:
                    paths.append(os.path.join(image_folder, f))
            paths.sort(key=lambda p: os.path.basename(p))
            annotated_count = sum(1 for p in paths if os.path.isfile(get_annotation_path(p)))
            cur.execute(
                "INSERT INTO episodes (run_id, name, image_count, annotated_count) VALUES (?, ?, ?, ?)",
                (run_id, ep_name, len(paths), annotated_count),
            )
            ep_id = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
            for p in paths:
                ann = 1 if os.path.isfile(get_annotation_path(p)) else 0
                rel = os.path.relpath(p, data_root).replace("\\", "/")
                filename = os.path.basename(p)
                sort_key = filename
                cur.execute(
                    "INSERT INTO images (episode_id, path_id, filename, annotated, sort_key) VALUES (?, ?, ?, ?, ?)",
                    (ep_id, rel, filename, ann, sort_key),
                )
            total_images += len(paths)
    conn.commit()
    conn.close()
    return {"runs": len(runs), "total_images": total_images}


def get_db_stats(db_path: str) -> dict:
    """库内统计：runs, episodes, images。用于部署校验。"""
    conn = sqlite3.connect(db_path)
    runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    return {"runs": runs, "episodes": episodes, "images": images}


def get_runs_from_db(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name FROM runs ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_episodes_from_db(db_path: str, run_name: str) -> list:
    conn = sqlite3.connect(db_path)
    run_id = conn.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()
    if not run_id:
        conn.close()
        return []
    run_id = run_id[0]
    rows = conn.execute(
        "SELECT name, image_count, annotated_count FROM episodes WHERE run_id = ? ORDER BY name",
        (run_id,),
    ).fetchall()
    conn.close()
    return [{"name": r[0], "imageCount": r[1], "annotatedCount": r[2]} for r in rows]


def get_images_page(db_path: str, page: int = 1, page_size: int = 50) -> list[dict]:
    """扁平分页：全表 images 按 id 排序，返回 [{ id, file_path, status }]。"""
    conn = sqlite3.connect(db_path)
    offset = (page - 1) * page_size
    rows = conn.execute(
        "SELECT path_id, annotated FROM images ORDER BY id ASC LIMIT ? OFFSET ?",
        (page_size, offset),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "file_path": r[0], "status": "annotated" if r[1] else "pending"}
        for r in rows
    ]


def get_images_cursor(db_path: str, run_name: str, ep_name: str, cursor: str | None, limit: int = 50) -> tuple[list, str | None]:
    """
    游标分页：WHERE sort_key > cursor ORDER BY sort_key LIMIT limit.
    返回 (items, next_cursor)。items 为 { id, filename, annotated }；next_cursor 为下一页游标或 None。
    """
    conn = sqlite3.connect(db_path)
    run_id = conn.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()
    if not run_id:
        conn.close()
        return [], None
    run_id = run_id[0]
    ep = conn.execute(
        "SELECT id FROM episodes WHERE run_id = ? AND name = ?", (run_id, ep_name)
    ).fetchone()
    if not ep:
        conn.close()
        return [], None
    ep_id = ep[0]
    if cursor is None:
        rows = conn.execute(
            "SELECT path_id, filename, annotated FROM images WHERE episode_id = ? ORDER BY sort_key ASC LIMIT ?",
            (ep_id, limit + 1),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT path_id, filename, annotated FROM images WHERE episode_id = ? AND sort_key > ? ORDER BY sort_key ASC LIMIT ?",
            (ep_id, cursor, limit + 1),
        ).fetchall()
    conn.close()
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [{"id": r[0], "filename": r[1], "annotated": bool(r[2])} for r in rows]
    next_cursor = rows[-1][1] if has_more and rows else None
    return items, next_cursor


# ---------- 标注工作台（SQLite 回退） ----------

_ANNOTATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    image_id INTEGER PRIMARY KEY REFERENCES images(id),
    boxes TEXT DEFAULT '[]',
    is_reviewed INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now')),
    user_id TEXT
);
"""


def _ensure_user_id_column(db_path: str) -> None:
    """确保 annotations 有 user_id 列（迁移旧库）"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("PRAGMA table_info(annotations)").fetchall()
    has_user_id = any(r[1] == "user_id" for r in rows)
    if not has_user_id:
        try:
            conn.execute("ALTER TABLE annotations ADD COLUMN user_id TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()


def _ensure_annotations_table(db_path: str) -> None:
    """确保 annotations 表存在（兼容旧 catalog.db）"""
    conn = sqlite3.connect(db_path)
    conn.executescript(_ANNOTATIONS_SCHEMA)
    conn.commit()
    conn.close()
    _ensure_user_id_column(db_path)


def get_image_by_path_id(db_path: str, path_id: str) -> dict | None:
    """根据 path_id 查询 images，返回 { id, path_id } 或 None"""
    _ensure_annotations_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT id, path_id FROM images WHERE path_id = ?", (path_id,)).fetchone()
    conn.close()
    return {"id": row[0], "path_id": row[1]} if row else None


def get_annotation_boxes(db_path: str, image_id: int) -> list:
    """查询 annotations 的 boxes"""
    import json
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT boxes FROM annotations WHERE image_id = ?", (image_id,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except Exception:
        return []


def upsert_annotation(db_path: str, image_id: int, boxes: list, user_id: str | None = None) -> None:
    """UPSERT annotations（含 user_id 以支持按用户统计）"""
    import json
    _ensure_user_id_column(db_path)
    boxes_json = json.dumps(boxes, ensure_ascii=False)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO annotations (image_id, boxes, is_reviewed, updated_at, user_id)
           VALUES (?, ?, 0, datetime('now'), ?)""",
        (image_id, boxes_json, user_id),
    )
    conn.commit()
    conn.close()


def get_all_annotated_for_export(db_path: str) -> list[dict]:
    """获取所有已标注图片：run, ep, path_id, filename, boxes。仅从 annotations 表（工作台保存）"""
    import json
    _ensure_annotations_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT r.name, e.name, i.path_id, i.filename, a.boxes
           FROM annotations a
           JOIN images i ON a.image_id = i.id
           JOIN episodes e ON i.episode_id = e.id
           JOIN runs r ON e.run_id = r.id
           ORDER BY r.name, e.name, i.sort_key"""
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        boxes = []
        if r[4]:
            try:
                boxes = json.loads(r[4]) if isinstance(r[4], str) else r[4]
            except Exception:
                pass
        out.append({"run": r[0], "ep": r[1], "path_id": r[2], "filename": r[3], "boxes": boxes})
    return out


def get_all_annotated_merged(db_path: str, data_root: str) -> list[dict]:
    """
    获取所有已标注图片（合并 DB + 磁盘）：run, ep, path_id, filename, boxes。
    DB annotations 表优先；若无则从 *_annot.json 读取，保证主标注页保存的标注也能导出与恢复。
    """
    import json
    _ensure_annotations_table(db_path)
    conn = sqlite3.connect(db_path)
    # 所有 annotated=1 的图片
    rows = conn.execute(
        """SELECT r.name, e.name, i.path_id, i.filename, i.id, a.boxes
           FROM images i
           JOIN episodes e ON i.episode_id = e.id
           JOIN runs r ON e.run_id = r.id
           LEFT JOIN annotations a ON a.image_id = i.id
           WHERE i.annotated = 1
           ORDER BY r.name, e.name, i.sort_key"""
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        run_name, ep_name, path_id, filename, img_id, db_boxes = r[0], r[1], r[2], r[3], r[4], r[5]
        boxes = []
        if db_boxes:
            try:
                boxes = json.loads(db_boxes) if isinstance(db_boxes, str) else db_boxes
            except Exception:
                pass
        if not boxes:
            # 从磁盘 *_annot.json 读取（主标注页保存的）
            full_path = os.path.normpath(os.path.join(data_root, path_id.replace("\\", "/")))
            if os.path.isfile(full_path):
                ann = load_annotation(full_path)
                objs = ann.get("objects") or ann.get("boxes") or []
                boxes = objs if isinstance(objs, list) else []
        out.append({"run": run_name, "ep": ep_name, "path_id": path_id, "filename": filename, "boxes": boxes})
    return out


def get_annotation_counts_by_user(db_path: str, run_name: str) -> list[dict]:
    """按用户统计某 run 的标注数量，返回 [{user_id, count}]。user_id 为空时表示历史数据。"""
    _ensure_user_id_column(db_path)
    conn = sqlite3.connect(db_path)
    run_row = conn.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()
    if not run_row:
        conn.close()
        return []
    run_id = run_row[0]
    rows = conn.execute(
        """SELECT COALESCE(a.user_id, '(历史)') AS uid, COUNT(*) AS cnt
           FROM annotations a
           JOIN images i ON a.image_id = i.id
           JOIN episodes e ON i.episode_id = e.id
           WHERE e.run_id = ?
           GROUP BY a.user_id""",
        (run_id,),
    ).fetchall()
    conn.close()
    return [{"user_id": r[0], "count": r[1]} for r in rows]


def update_image_annotated(db_path: str, image_id: int, annotated: bool = True) -> None:
    """更新 images.annotated 并刷新对应 episode 的 annotated_count"""
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE images SET annotated = ? WHERE id = ?", (1 if annotated else 0, image_id))
    conn.execute(
        """UPDATE episodes SET annotated_count = (
            SELECT COUNT(*) FROM images WHERE episode_id = episodes.id AND annotated = 1
        ) WHERE id = (SELECT episode_id FROM images WHERE id = ?)""",
        (image_id,),
    )
    conn.commit()
    conn.close()
