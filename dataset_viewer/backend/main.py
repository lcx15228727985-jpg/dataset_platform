"""
FastAPI 后端：仅负责「给图」和「存标注」，不参与渲染。
读取逻辑优先查库：有 DATABASE_URL 时用 PostgreSQL，否则用 SQLite 或文件系统遍历。
"""
from pathlib import Path

from dotenv import load_dotenv

# Load .env from dataset_viewer/ before auth/mongo_db read JWT_SECRET, MONGODB_URI
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncio
import io
import json
import os
import tempfile
import zipfile
from collections import OrderedDict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import (
    create_token,
    get_current_user,
    require_user,
    verify_user_async,
)
from .data import (
    get_data_root,
    get_run_folders,
    list_ep_folders,
    count_images,
    count_images_on_disk,
    count_annotated,
    is_annotated,
    load_annotation,
    save_annotation,
    delete_annotation,
    image_path_to_id,
    image_id_to_path,
)

app = FastAPI(title="数据集标注平台 API")


@app.on_event("startup")
async def startup():
    """启动时初始化 MongoDB 用户（若有 MONGODB_URI）"""
    try:
        from .mongo_db import ensure_users_seeded, get_client
        client = await get_client()
        if client is not None:
            ok = await ensure_users_seeded()
            if ok:
                pass  # MongoDB 用户已就绪
    except Exception:
        pass


@app.on_event("shutdown")
async def shutdown():
    if pg_db is not None:
        await pg_db.close_pool()
    try:
        from .mongo_db import close_client
        close_client()
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = get_data_root()

# 分页/列表上限，避免单次请求过大导致 CPU 与内存飙升
MAX_PAGE_SIZE = 100
MAX_EPISODE_IMAGES_LIMIT = 100

# 缩略图缓存：同一张图重复请求时直接返回缓存，降低 CPU（标注量大时大量缩略图请求会占满 CPU）
_THUMB_CACHE: OrderedDict = OrderedDict()
_THUMB_CACHE_MAX = 500

# SQLite 回退
try:
    from .db import (
        get_db_path,
        get_db_stats,
        get_runs_from_db,
        get_episodes_from_db,
        get_images_cursor,
        get_images_page,
        get_image_by_path_id,
        get_annotation_boxes,
        upsert_annotation,
        update_image_annotated,
        get_all_annotated_for_export,
        get_annotation_counts_by_user,
    )
    _db_path = get_db_path()
    USE_DB = os.path.isfile(_db_path)
except Exception:
    USE_DB = False
    _db_path = None
    get_db_path = get_db_stats = get_runs_from_db = get_episodes_from_db = get_images_cursor = get_images_page = None  # type: ignore
    get_image_by_path_id = get_annotation_boxes = upsert_annotation = update_image_annotated = None  # type: ignore
    get_all_annotated_for_export = get_annotation_counts_by_user = None  # type: ignore

try:
    from . import pg_db
except Exception:
    pg_db = None  # type: ignore


# ---------- 数据模型 ----------
class BoxItem(BaseModel):
    id: str = ""
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    label: str = ""


class WorkstationSavePayload(BaseModel):
    path_id: str
    boxes: list[BoxItem | dict]


class LoginPayload(BaseModel):
    user_id: str
    password: str


class AnnotationBody(BaseModel):
    labels: list = []
    objects: list = []
    canvas_width: int | None = None
    canvas_height: int | None = None
    image_width: int | None = None
    image_height: int | None = None


# ---------- 认证 API ----------


@app.post("/api/auth/login")
async def api_auth_login(payload: LoginPayload):
    """登录：user_id 000-009，密码 123456。返回 JWT。"""
    if not payload.user_id or not payload.password:
        raise HTTPException(status_code=400, detail="请输入账号和密码")
    uid = payload.user_id.strip()
    user = await verify_user_async(uid, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="账号不存在或密码错误")
    role = user.get("role", "root")
    token = create_token(uid, role=role)
    return {"token": token, "user_id": uid, "role": role}


@app.get("/api/auth/me")
async def api_auth_me(current: dict = Depends(get_current_user)):
    """验证当前 token 是否有效，返回当前用户。"""
    if not current:
        raise HTTPException(status_code=401, detail="未登录或 token 已过期")
    return {"user_id": current["user_id"], "role": current["role"]}


# ---------- API（读取统一查库：优先 Postgres，否则 SQLite，否则文件系统） ----------


async def _use_pg():
    """是否使用 Postgres（有 DATABASE_URL 且连接成功）"""
    if pg_db is None:
        return False
    pool = await pg_db.get_pool()
    return pool is not None


@app.get("/api/health")
async def api_health():
    """健康检查与部署状态：是否使用数据库、数据根目录等（便于 Linux 部署排查）"""
    use_pg = await _use_pg()
    return {
        "ok": True,
        "database": "postgres" if use_pg else ("sqlite" if (USE_DB and _db_path) else "none"),
        "data_root": DATA_ROOT,
        "data_root_exists": os.path.isdir(DATA_ROOT),
    }


@app.get("/api/deploy/status")
async def api_deploy_status(verify: bool = False):
    """
    部署与入库情况：库内 runs/episodes/images 统计。
    verify=1 时同时扫描 DATA_ROOT 磁盘，对比是否全部入库（db.images == disk.total_images）。
    """
    use_pg = await _use_pg()
    db_stats = {"runs": 0, "episodes": 0, "images": 0}
    if use_pg and pg_db:
        db_stats = await pg_db.get_db_stats()
    elif USE_DB and _db_path:
        db_stats = await asyncio.to_thread(get_db_stats, _db_path)

    out = {
        "database": "postgres" if use_pg else ("sqlite" if (USE_DB and _db_path) else "none"),
        "data_root": DATA_ROOT,
        "data_root_exists": os.path.isdir(DATA_ROOT),
        "db": db_stats,
    }

    if verify:
        disk = count_images_on_disk(DATA_ROOT)
        out["disk"] = disk
        out["consistent"] = (
            out["data_root_exists"]
            and db_stats["images"] == disk["total_images"]
            and disk["total_images"] >= 0
        )
        out["message"] = (
            "图片信息已全部入库"
            if out.get("consistent") is True
            else (
                "库内与磁盘数量不一致，请重新执行 init_pg_db 或 scan_dataset"
                if out["data_root_exists"] and db_stats["images"] != disk["total_images"]
                else "数据根目录不存在或磁盘无图片"
            )
        )

    return out


@app.get("/api/dashboard/user-stats")
async def api_dashboard_user_stats(
    run: str,
    current: dict | None = Depends(get_current_user),
):
    """
    看板按用户统计：当前用户标注数量、其他用户标注数量。
    需有 catalog DB。未登录时 current_user_count 为 0。
    """
    if not await _workstation_available():
        return {"current_user_id": None, "current_user_count": 0, "others": []}
    current_uid = current.get("user_id") if current else None
    others = []
    if await _use_pg() and pg_db:
        rows = await pg_db.get_annotation_counts_by_user(run)
    elif USE_DB and get_annotation_counts_by_user and _db_path:
        rows = await asyncio.to_thread(get_annotation_counts_by_user, _db_path, run)
    else:
        rows = []
    current_count = 0
    for r in rows:
        uid = r.get("user_id", "")
        cnt = r.get("count", 0)
        if uid == current_uid:
            current_count = cnt
        else:
            others.append({"user_id": uid, "count": cnt})
    return {
        "current_user_id": current_uid,
        "current_user_count": current_count,
        "others": others,
    }


@app.get("/api/runs")
async def api_runs():
    """返回 run 文件夹列表（有 DB 时走索引）；无数据时返回空列表不报错"""
    if await _use_pg():
        runs = await pg_db.get_runs()
        return {"runs": runs or [], "cursorAvailable": True}
    if USE_DB and _db_path:
        runs = await asyncio.to_thread(get_runs_from_db, _db_path)
        return {"runs": runs or [], "cursorAvailable": True}
    return {"runs": get_run_folders(DATA_ROOT) or [], "cursorAvailable": False}


@app.get("/api/runs/{run}/episodes")
async def api_episodes(run: str):
    """返回该 run 下各 episode 摘要。run 为空或不存在时返回空列表"""
    if not (run or run.strip()):
        return {"episodes": [], "cursorAvailable": False}
    if await _use_pg():
        eps = await pg_db.get_episodes(run)
        return {"episodes": eps, "cursorAvailable": True}
    if USE_DB and _db_path:
        eps = await asyncio.to_thread(get_episodes_from_db, _db_path, run)
        return {"episodes": eps, "cursorAvailable": True}
    ep_list = list_ep_folders(DATA_ROOT, run)
    out = []
    for name, image_folder in ep_list:
        if not image_folder:
            out.append({"name": name, "imageCount": 0, "annotatedCount": 0, "images": []})
            continue
        total, paths = count_images(image_folder)
        ann_count = count_annotated(paths)
        images = []
        for p in paths:
            rel_id = image_path_to_id(DATA_ROOT, p)
            images.append({
                "id": rel_id,
                "filename": os.path.basename(p),
                "annotated": is_annotated(p),
            })
        out.append({
            "name": name,
            "imageCount": total,
            "annotatedCount": ann_count,
            "images": images,
        })
    return {"episodes": out, "cursorAvailable": False}


@app.get("/api/images")
async def api_images(page: int = 1, page_size: int = 50):
    """扁平分页：从库中读取图片列表，返回 id, file_path, status。需已建库；page_size 有上限以控制 CPU/内存。"""
    page_size = min(max(1, page_size), MAX_PAGE_SIZE)
    if await _use_pg():
        rows = await pg_db.get_images_page(page, page_size)
        return rows
    if USE_DB and _db_path:
        rows = await asyncio.to_thread(get_images_page, _db_path, page, page_size)
        return rows
    raise HTTPException(
        status_code=501,
        detail="Images list requires catalog DB. Run: python -m backend.scan_dataset or python -m backend.init_pg_db",
    )


@app.get("/api/runs/{run}/episodes/{ep}/images")
async def api_episode_images(run: str, ep: str, cursor: str | None = None, limit: int = 50):
    """游标分页：从库中拉取该 episode 的图片。需已建库；limit 有上限以控制 CPU/内存。"""
    limit = min(max(1, limit), MAX_EPISODE_IMAGES_LIMIT)
    if await _use_pg():
        items, next_cursor = await pg_db.get_images_cursor(run, ep, cursor, limit)
        return {"items": items, "nextCursor": next_cursor}
    if USE_DB and _db_path:
        items, next_cursor = await asyncio.to_thread(get_images_cursor, _db_path, run, ep, cursor, limit)
        return {"items": items, "nextCursor": next_cursor}
    raise HTTPException(
        status_code=501,
        detail="Cursor API requires catalog DB. Run: python -m backend.scan_dataset or python -m backend.init_pg_db",
    )


def _media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    m = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff"}
    return m.get(ext, "application/octet-stream")


def _serve_image(image_id_or_path: str, thumb: bool = False, thumb_size: int = 150):
    """解析 id/path 并返回图片文件；thumb=True 时返回缩略图（最长边 thumb_size px），带 LRU 缓存以降低 CPU"""
    import urllib.parse
    raw = urllib.parse.unquote(image_id_or_path)
    path = image_id_to_path(DATA_ROOT, raw)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not thumb:
        return FileResponse(path, media_type=_media_type(path))
    cache_key = (raw, thumb_size)
    if cache_key in _THUMB_CACHE:
        _THUMB_CACHE.move_to_end(cache_key)
        return Response(content=_THUMB_CACHE[cache_key], media_type="image/jpeg")
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > thumb_size or h > thumb_size:
            r = min(thumb_size / w, thumb_size / h)
            img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        while len(_THUMB_CACHE) >= _THUMB_CACHE_MAX:
            _THUMB_CACHE.popitem(last=False)
        _THUMB_CACHE[cache_key] = data
        return Response(content=data, media_type="image/jpeg")
    except Exception:
        return FileResponse(path, media_type=_media_type(path))


@app.get("/api/image")
def api_get_image_query(path: str = "", thumb: int = 0):
    """path= 相对 data root；thumb=1 时返回缩略图（最长边 150px）"""
    if not path:
        raise HTTPException(status_code=400, detail="Missing path parameter")
    return _serve_image(path, thumb=thumb == 1)


@app.get("/api/image/{image_id:path}")
def api_get_image_path(image_id: str):
    """根据 URL 路径形式返回图片（备用）"""
    return _serve_image(image_id)


# ---------- 标注工作台 API（PostgreSQL 优先，SQLite 回退） ----------


def _get_image_dimensions(file_path: str) -> tuple[int, int]:
    """返回 (width, height)，失败返回 (0, 0)"""
    if not file_path or not os.path.isfile(file_path):
        return 0, 0
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


async def _workstation_available() -> bool:
    """工作台可用：Postgres 或 SQLite"""
    if await _use_pg() and pg_db:
        return True
    return bool(USE_DB and _db_path and get_image_by_path_id is not None)


@app.get("/api/workstation/data")
async def api_workstation_data(path: str = ""):
    """
    获取工作台数据：图片 path、尺寸、已有 boxes。
    path: 图片 path_id（相对 data root）。
    支持 PostgreSQL 或 SQLite（scan_dataset 后可用）。
    """
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Missing path parameter")
    if not await _workstation_available():
        raise HTTPException(
            status_code=501,
            detail="Workstation API requires catalog DB. Run: python -m backend.init_pg_db or python -m backend.scan_dataset",
        )
    path = path.strip()
    file_path = image_id_to_path(DATA_ROOT, path)
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if await _use_pg() and pg_db:
        img_row = await pg_db.get_image_by_path_id(path)
        if not img_row:
            raise HTTPException(status_code=404, detail="Image not in catalog")
        boxes = await pg_db.get_annotation_boxes(img_row["id"])
    else:
        img_row = await asyncio.to_thread(get_image_by_path_id, _db_path, path)
        if not img_row:
            raise HTTPException(status_code=404, detail="Image not in catalog")
        boxes = await asyncio.to_thread(get_annotation_boxes, _db_path, img_row["id"])
    width, height = _get_image_dimensions(file_path)
    return {
        "path_id": path,
        "meta": {"width": width, "height": height},
        "boxes": boxes,
    }


@app.post("/api/workstation/save")
async def api_workstation_save(
    payload: WorkstationSavePayload,
    current: dict = Depends(require_user),
):
    """
    保存标注 boxes（UPSERT）。需登录，仅登录用户的标注会写入数据库。
    支持 PostgreSQL 或 SQLite。
    """
    if not await _workstation_available():
        raise HTTPException(
            status_code=501,
            detail="Workstation API requires catalog DB. Run: python -m backend.scan_dataset",
        )
    user_id = current.get("user_id", "")
    if await _use_pg() and pg_db:
        img_row = await pg_db.get_image_by_path_id(payload.path_id.strip())
        if not img_row:
            raise HTTPException(status_code=404, detail="Image not in catalog")
        boxes = [b.model_dump() if not isinstance(b, dict) else b for b in payload.boxes]
        await pg_db.upsert_annotation(img_row["id"], boxes, user_id)
        await pg_db.update_image_annotated(img_row["id"], annotated=len(boxes) > 0)
    else:
        img_row = await asyncio.to_thread(get_image_by_path_id, _db_path, payload.path_id.strip())
        if not img_row:
            raise HTTPException(status_code=404, detail="Image not in catalog")
        boxes = [b.model_dump() if not isinstance(b, dict) else b for b in payload.boxes]
        await asyncio.to_thread(upsert_annotation, _db_path, img_row["id"], boxes, user_id)
        await asyncio.to_thread(update_image_annotated, _db_path, img_row["id"], annotated=len(boxes) > 0)
    return {"status": "success"}


# ---------- 导出已标注数据 ----------


def _get_export_dir() -> str:
    """导出目录：EXPORT_DIR 或 DATA_ROOT/export"""
    export_dir = os.environ.get("EXPORT_DIR")
    if export_dir and os.path.isabs(export_dir):
        return export_dir
    return os.path.join(DATA_ROOT, "export")


@app.get("/api/export/annotations")
async def api_export_annotations(format: str = "json", save: bool = False):
    """
    导出所有已标注图片数据：run、ep、path_id、filename、boxes。
    format=json（默认）或 csv。save=1 时同时保存到服务器导出目录（用于 Linux 训练）。
    """
    if not await _workstation_available() or get_all_annotated_for_export is None:
        raise HTTPException(
            status_code=501,
            detail="Export requires catalog DB. Run: python -m backend.scan_dataset or init_pg_db",
        )
    if await _use_pg() and pg_db:
        rows = await pg_db.get_all_annotated_for_export()
    else:
        rows = await asyncio.to_thread(get_all_annotated_for_export, _db_path)
    fmt = (format or "json").lower()
    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["run", "ep", "path_id", "filename", "boxes"])
        for r in rows:
            w.writerow([r["run"], r["ep"], r["path_id"], r["filename"], json.dumps(r["boxes"], ensure_ascii=False)])
        content = buf.getvalue().encode("utf-8-sig")
        media_type = "text/csv; charset=utf-8"
        filename = "annotations.csv"
    else:
        content = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        media_type = "application/json"
        filename = "annotations.json"
    if save:
        try:
            export_dir = _get_export_dir()
            os.makedirs(export_dir, exist_ok=True)
            out_path = os.path.join(export_dir, filename)
            with open(out_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/annotated-images")
async def api_export_annotated_images_zip():
    """
    导出所有已标注图片的 PNG 为 ZIP。
    目录结构：{run}标注版/episode_X/xxx.png，便于查看哪些已标注。
    """
    try:
        if not await _workstation_available() or get_all_annotated_for_export is None:
            raise HTTPException(
                status_code=501,
                detail="Export requires catalog DB. Run: python -m backend.scan_dataset or init_pg_db",
            )
        if await _use_pg() and pg_db:
            rows = await pg_db.get_all_annotated_for_export()
        else:
            rows = await asyncio.to_thread(get_all_annotated_for_export, _db_path)
        if not rows:
            raise HTTPException(status_code=404, detail="No annotated images to export")

        def draw_boxes_on_image(img_path: str, boxes: list) -> bytes | None:
            """在图片上绘制标注框（椭圆/矩形），返回 PNG 字节。boxes 为归一化坐标 x,y,w,h (0~1)。"""
            try:
                from PIL import Image, ImageDraw
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                draw = ImageDraw.Draw(img)
                outline_color = (255, 0, 0)  # 红色
                line_width = max(2, min(w, h) // 300)
                for b in boxes or []:
                    if not isinstance(b, dict):
                        continue
                    x = float(b.get("x", 0))
                    y = float(b.get("y", 0))
                    bw = float(b.get("w", 0.1))
                    bh = float(b.get("h", 0.1))
                    x1 = x * w
                    y1 = y * h
                    x2 = (x + bw) * w
                    y2 = (y + bh) * h
                    if b.get("type") == "rect":
                        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=line_width)
                    else:
                        draw.ellipse([x1, y1, x2, y2], outline=outline_color, width=line_width)
                out = io.BytesIO()
                img.save(out, format="PNG")
                out.seek(0)
                return out.getvalue()
            except Exception:
                return None

        def build_zip_to_path(zip_path: str) -> None:
            """将 ZIP 写入文件，使用低压缩级别以加快导出、减少 CPU；避免整包进内存。"""
            with zipfile.ZipFile(
                zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1
            ) as zf:
                for r in rows:
                    run_name = r.get("run", "")
                    ep_name = r.get("ep", "")
                    path_id = (r.get("path_id") or "").replace("\\", "/").strip()
                    filename = r.get("filename") or os.path.basename(path_id)
                    full_path = os.path.normpath(os.path.join(DATA_ROOT, path_id))
                    if not os.path.isfile(full_path):
                        continue
                    arcname = f"{run_name}标注版/{ep_name}/{filename}"
                    boxes = r.get("boxes") or []
                    if boxes:
                        png_bytes = draw_boxes_on_image(full_path, boxes)
                        if png_bytes:
                            zf.writestr(arcname, png_bytes)
                        else:
                            try:
                                zf.write(full_path, arcname=arcname)
                            except Exception:
                                continue
                    else:
                        try:
                            zf.write(full_path, arcname=arcname)
                        except Exception:
                            continue

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.close()
        try:
            await asyncio.to_thread(build_zip_to_path, tmp.name)
        except Exception:
            if os.path.isfile(tmp.name):
                os.unlink(tmp.name)
            raise
        zip_filename = f"{rows[0].get('run', 'annotated')}标注版.zip"
        from urllib.parse import quote
        safe_filename = quote(zip_filename, safe="")
        disposition = f'attachment; filename="annotated_images.zip"; filename*=UTF-8\'\'{safe_filename}'
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            headers={"Content-Disposition": disposition},
            background=BackgroundTask(lambda: os.path.isfile(tmp.name) and os.unlink(tmp.name)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 旧版标注 API（文件存储，保留兼容） ----------


@app.get("/api/annotation/{image_id:path}")
def api_get_annotation(image_id: str):
    """获取某张图的标注 JSON"""
    path = image_id_to_path(DATA_ROOT, image_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    data = load_annotation(path)
    return data


@app.post("/api/annotation/{image_id:path}")
def api_save_annotation(image_id: str, body: AnnotationBody):
    """保存标注"""
    path = image_id_to_path(DATA_ROOT, image_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    data = body.model_dump()
    save_annotation(path, data)
    return {"status": "success"}


@app.delete("/api/annotation/{image_id:path}")
async def api_delete_annotation(image_id: str):
    """删除标注（文件 + 数据库，保证主页与工作台同步）"""
    path = image_id_to_path(DATA_ROOT, image_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    delete_annotation(path)
    if await _workstation_available():
        if await _use_pg() and pg_db:
            img_row = await pg_db.get_image_by_path_id(image_id.strip())
            if img_row:
                await pg_db.upsert_annotation(img_row["id"], [])
                await pg_db.update_image_annotated(img_row["id"], annotated=False)
        else:
            img_row = await asyncio.to_thread(get_image_by_path_id, _db_path, image_id.strip())
            if img_row:
                await asyncio.to_thread(upsert_annotation, _db_path, img_row["id"], [])
                await asyncio.to_thread(update_image_annotated, _db_path, img_row["id"], annotated=False)
    return {"status": "success"}


# ---------- 生产环境：挂载前端静态文件（STATIC_DIR 由 Docker 设置） ----------

STATIC_DIR = os.environ.get("STATIC_DIR")


@app.exception_handler(404)
async def custom_404(request, exc):
    """SPA 路由回退：非 /api 的 404 返回 index.html"""
    if request.url.path.startswith("/api"):
        raise exc
    if STATIC_DIR and os.path.isdir(STATIC_DIR):
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
    raise exc


if STATIC_DIR and os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
