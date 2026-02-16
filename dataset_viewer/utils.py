"""
共享逻辑：数据路径、标注读写、画布坐标与绘图。
仅被 app 与 views 使用，禁止 views 引用 app。
"""
import json
import os

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# 路径与常量（数据根目录由 app 传入，此处仅定义结构常量）
IMAGE_SUBFOLDER = "images"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
IMAGES_PNG_SUBDIR = "images_png"


def get_run_folders(root):
    """返回 root 下所有子文件夹名（作为 run 候选）"""
    if not os.path.isdir(root):
        return []
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def list_ep_folders(root, run_folder):
    """返回 [(ep0, images_path), ...]，优先 images_png"""
    result = []
    run_path = os.path.join(root, run_folder) if run_folder else root
    for i in range(13):
        name = f"ep{i}"
        episode_dir = os.path.join(run_path, f"episode_{i}")
        image_folder = os.path.join(episode_dir, IMAGES_PNG_SUBDIR)
        if not os.path.isdir(image_folder):
            image_folder = os.path.join(episode_dir, IMAGE_SUBFOLDER)
        if os.path.isdir(image_folder):
            result.append((name, image_folder))
        else:
            result.append((name, None))
    return result


def count_images(folder_path):
    """统计文件夹内图片数量并返回图片路径列表（按文件名排序）"""
    if not folder_path or not os.path.isdir(folder_path):
        return 0, []
    paths = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXT:
            paths.append(os.path.join(folder_path, f))
    paths.sort(key=lambda p: os.path.basename(p))
    return len(paths), paths


def get_annotation_path(image_path):
    """返回与该图片同目录的标注 JSON 路径"""
    return os.path.splitext(image_path)[0] + "_annot.json"


def is_annotated(image_path):
    """判断图片是否已有标注"""
    return os.path.isfile(get_annotation_path(image_path))


def count_annotated(paths):
    """统计路径列表中已标注的数量"""
    return sum(1 for p in paths if is_annotated(p))


def load_annotation(image_path):
    """加载已有标注（JSON）"""
    p = get_annotation_path(image_path)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"labels": [], "objects": []}


def save_annotation(image_path, data):
    """保存标注到 JSON"""
    p = get_annotation_path(image_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_annotation(image_path):
    """删除标注（移除 _annot.json）"""
    p = get_annotation_path(image_path)
    if os.path.isfile(p):
        os.remove(p)


def clear_canvas_state_for_path(image_path):
    """取消标注后清除该图在 session 中的画布状态"""
    import streamlit as st
    key_objs = "anno_canvas_objs_" + image_path.replace(os.sep, "_")
    key_reset = key_objs + "_reset"
    for k in (key_objs, key_reset):
        if k in st.session_state:
            del st.session_state[k]


def _obj_to_corner_bbox(o, canvas_w, canvas_h, img_w, img_h):
    """将 Fabric 对象坐标转为 (left, top, width, height) 图像坐标"""
    left = float(o.get("left", 0))
    top = float(o.get("top", 0))
    w = float(o.get("width", 0))
    h = float(o.get("height", 0))
    scale_x = float(o.get("scaleX", 1))
    scale_y = float(o.get("scaleY", 1))
    w_eff = w * scale_x
    h_eff = h * scale_y
    if o.get("type") in ("ellipse", "circle"):
        ox = o.get("originX", "left")
        oy = o.get("originY", "top")
    else:
        ox, oy = o.get("originX", "left"), o.get("originY", "top")
    if ox == "center":
        left = left - w_eff / 2
    if oy == "center":
        top = top - h_eff / 2
    if canvas_w and canvas_h and img_w and img_h:
        sx, sy = img_w / canvas_w, img_h / canvas_h
        left, top, w_eff, h_eff = left * sx, top * sy, w_eff * sx, h_eff * sy
    return (left, top, w_eff, h_eff)


def draw_shadow_objects_on_image(pil_img, objs, canvas_w=None, canvas_h=None, color="#999999"):
    """在图片上绘制已有标注框的阴影"""
    if not objs or not PILImage:
        return pil_img
    from PIL import ImageDraw
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    cw = canvas_w or img_w
    ch = canvas_h or img_h
    for o in objs:
        l, t, ww, hh = _obj_to_corner_bbox(o, cw, ch, img_w, img_h)
        if o.get("type") in ("ellipse", "circle"):
            draw.ellipse([l, t, l + ww, t + hh], outline=color, width=2)
        else:
            draw.rectangle([l, t, l + ww, t + hh], outline=color, width=2)
    return img


def draw_grid_on_image(pil_img, grid_size=50, color="#888888"):
    """在图片上绘制网格线"""
    if not PILImage:
        return pil_img
    from PIL import ImageDraw
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w + 1, grid_size):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h + 1, grid_size):
        draw.line([(0, y), (w, y)], fill=color, width=1)
    return img


def render_objects_on_image(pil_img, objs, canvas_w, canvas_h):
    """将标注框绘制到图片上（保存前预览），返回 PIL Image"""
    if not objs or not PILImage:
        return pil_img
    img = pil_img.copy()
    img_w, img_h = img.size
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for o in objs:
            l, t, w, h = _obj_to_corner_bbox(o, canvas_w, canvas_h, img_w, img_h)
            l, t = int(round(l)), int(round(t))
            w, h = int(round(w)), int(round(h))
            if o.get("type") in ("ellipse", "circle"):
                draw.ellipse([l, t, l + w, t + h], outline="#00FF00", width=2)
            else:
                draw.rectangle([l, t, l + w, t + h], outline="#00FF00", width=2)
    except Exception:
        pass
    return img


def render_image_with_boxes(image_path):
    """将标注框绘制到图片上，返回 PIL Image 或 None"""
    if not PILImage or not is_annotated(image_path):
        return None
    try:
        img = PILImage.open(image_path).convert("RGB")
    except Exception:
        return None
    ann = load_annotation(image_path)
    objs = ann.get("objects", [])
    if not objs:
        return img
    img_w, img_h = img.size
    cw = ann.get("canvas_width") or img_w
    ch = ann.get("canvas_height") or img_h
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for o in objs:
            l, t, w, h = _obj_to_corner_bbox(o, cw, ch, img_w, img_h)
            if o.get("type") in ("ellipse", "circle"):
                draw.ellipse([l, t, l + w, t + h], outline="#00FF00", width=3)
            else:
                draw.rectangle([l, t, l + w, t + h], outline="#00FF00", width=3)
    except Exception:
        pass
    return img


def convert_folder_to_png(folder_path):
    """将文件夹内所有非 PNG 转为 PNG，返回 (转换数量, 错误列表)"""
    if not folder_path or not os.path.isdir(folder_path) or not PILImage:
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
            img = PILImage.open(path).convert("RGB")
            img.save(png_path, "PNG")
            os.remove(path)
            converted += 1
        except Exception as e:
            errors.append(f"{f}: {e}")
    return converted, errors
