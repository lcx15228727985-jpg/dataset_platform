"""
曲率数据集可视化平台 · ep0 ~ ep12
展示每个 ep 的数量，并可展开预览该 ep 下全部图片。
点击预览图旁的「进入标注」即可进入标注工作台。
"""
import os
# 避免本机 SSL/OpenMP 导致无法启动
os.environ.pop("SSLKEYLOGFILE", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

# 必须最先调用
st.set_page_config(page_title="数据集标注平台", layout="wide")

# 标注工作台 session state
if "annotation_image_path" not in st.session_state:
    st.session_state.annotation_image_path = None
if "annotation_ep_name" not in st.session_state:
    st.session_state.annotation_ep_name = None
if "annotation_frame_name" not in st.session_state:
    st.session_state.annotation_frame_name = None
if "annotation_image_paths" not in st.session_state:
    st.session_state.annotation_image_paths = []
if "annotation_image_index" not in st.session_state:
    st.session_state.annotation_image_index = 0
if "annotation_saved_msg" not in st.session_state:
    st.session_state.annotation_saved_msg = None  # 用于显示「已保存」提示

# 数据根目录：与 dataset_viewer 同级的 dataset 文件夹（可通过环境变量 DATA_ROOT 覆盖）
_APP_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.normpath(os.path.join(_APP_DIR, "..", "dataset"))

# 路径结构: dataset/<run>/episode_<N>/images  （例如 dataset0209/episode_0/images）
IMAGE_SUBFOLDER = "images"  # 每个 episode_N 下图片所在子文件夹名
# 支持的图片后缀
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}


def get_run_folders(root):
    """返回 root 下所有子文件夹名（作为 run 候选，如 dataset0209）"""
    if not os.path.isdir(root):
        return []
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


# 优先使用 Zarr 导出后的 PNG 目录，否则用原始 images
IMAGES_PNG_SUBDIR = "images_png"


def list_ep_folders(root, run_folder):
    """
    路径结构: root/run_folder/episode_0/images 或 episode_0/images_png（导出后）
    返回 [(ep0, images_path), (ep1, images_path), ...]，优先 images_png
    """
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
    import json
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
    import json
    p = get_annotation_path(image_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_annotation(image_path):
    """删除标注（移除 _annot.json），图片保留"""
    p = get_annotation_path(image_path)
    if os.path.isfile(p):
        os.remove(p)


def _obj_to_corner_bbox(o, canvas_w, canvas_h, img_w, img_h):
    """
    将 Fabric.js 对象坐标转为 PIL 绘制用的 (corner_left, corner_top, width, height) 图像坐标。
    streamlit-drawable-canvas 椭圆：若未标注 origin 则视为 corner（左上角），否则按 origin 转换。
    """
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
        if ox == "center":
            left = left - w_eff / 2
        if oy == "center":
            top = top - h_eff / 2
    if canvas_w and canvas_h and img_w and img_h:
        sx, sy = img_w / canvas_w, img_h / canvas_h
        left, top, w_eff, h_eff = left * sx, top * sy, w_eff * sx, h_eff * sy
    return (left, top, w_eff, h_eff)


def draw_shadow_objects_on_image(pil_img, objs, canvas_w=None, canvas_h=None, color="#999999"):
    """在图片上绘制已有标注框的阴影（半透明灰），作为相对移动时的参考"""
    if not objs:
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
    """在图片上绘制网格线，便于框选时定位、避免偏置"""
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
    """将标注框绘制到图片上（用于工作台内保存前预览），返回 PIL Image"""
    if not objs or not PILImage:
        return pil_img
    img = pil_img.copy()
    img_w, img_h = img.size
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for o in objs:
            l, t, w, h = _obj_to_corner_bbox(o, canvas_w, canvas_h, img_w, img_h)
            if o.get("type") in ("ellipse", "circle"):
                draw.ellipse([l, t, l + w, t + h], outline="#00FF00", width=3)
            else:
                draw.rectangle([l, t, l + w, t + h], outline="#00FF00", width=3)
    except Exception:
        pass
    return img


def render_image_with_boxes(image_path):
    """将标注框绘制到图片上，返回 PIL Image（框选标记留在图片上）"""
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
    """
    将文件夹内所有非 PNG 图片转为 PNG（覆盖为同主名 .png 并删除原文件）。
    返回 (转换数量, 错误列表)。
    """
    if not folder_path or not os.path.isdir(folder_path):
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


def render_annotation_workspace():
    """标注工作台：大图 + 拉框标注 + 确认/保存 + 上一张/下一张"""
    path = st.session_state.annotation_image_path
    ep_name = st.session_state.annotation_ep_name or ""
    paths_list = st.session_state.annotation_image_paths
    idx = st.session_state.annotation_image_index
    frame_name = st.session_state.annotation_frame_name or (os.path.basename(path) if path else "")

    if not path or not os.path.isfile(path):
        st.session_state.annotation_image_path = None
        st.rerun()
        return

    st.subheader("📝 标注工作台")
    n_total = len(paths_list)
    n_annotated = count_annotated(paths_list)
    st.caption(f"**{ep_name}** · {frame_name} · 第 {idx + 1}/{n_total} 张 · 本 ep 已标注 {n_annotated}/{n_total} 张")

    # 工作台内进度条：已保存 / 本 ep 进度
    prog_val = n_annotated / n_total if n_total else 0
    st.progress(prog_val, text=f"本 ep 标注进度：{n_annotated}/{n_total} 已保存")
    if st.session_state.annotation_saved_msg:
        st.success(st.session_state.annotation_saved_msg)
        st.session_state.annotation_saved_msg = None

    # 操作按钮行：返回 | 上一张 | 下一张
    col_ret, col_prev, col_next, _ = st.columns([1, 1, 1, 5])
    with col_ret:
        if st.button("← 返回图库", key="btn_ret"):
            st.session_state.annotation_image_path = None
            st.session_state.annotation_ep_name = None
            st.session_state.annotation_frame_name = None
            st.session_state.annotation_image_paths = []
            st.session_state.annotation_image_index = 0
            st.session_state.annotation_saved_msg = None
            st.rerun()
            return
    with col_prev:
        if idx > 0 and st.button("← 上一张", key="btn_prev"):
            st.session_state.annotation_image_path = paths_list[idx - 1]
            st.session_state.annotation_frame_name = os.path.basename(paths_list[idx - 1])
            st.session_state.annotation_image_index = idx - 1
            st.session_state.annotation_saved_msg = None
            st.rerun()
    with col_next:
        if idx < n_total - 1 and st.button("下一张 →", key="btn_next"):
            st.session_state.annotation_image_path = paths_list[idx + 1]
            st.session_state.annotation_frame_name = os.path.basename(paths_list[idx + 1])
            st.session_state.annotation_image_index = idx + 1
            st.session_state.annotation_saved_msg = None
            st.rerun()

    ann_data = load_annotation(path)
    label_input = st.sidebar.text_input("标注标签", value="", key="anno_label")
    if st.sidebar.button("添加标签"):
        if label_input.strip():
            ann_data["labels"].append(label_input.strip())
            save_annotation(path, ann_data)
            st.session_state.annotation_saved_msg = "✅ 已保存（标签）"
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("已有标签:")
    for i, lb in enumerate(ann_data.get("labels", [])):
        st.sidebar.text(f"  • {lb}")

    try:
        img = PILImage.open(path).convert("RGB")
    except Exception as e:
        st.error(f"无法加载图片: {e}")
        return

    img_w, img_h = img.size
    # 显示缩放：侧边栏可调，使操作区+预览区合理适配屏幕
    max_side = st.sidebar.slider(
        "显示最大边长(px)", min_value=400, max_value=1200, value=600, step=50,
        key="anno_max_side",
        help="操作区与预览区共用此值，缩小可改善双栏布局"
    )
    scale = min(1.0, max_side / max(img_w, img_h))
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)

    # 网格线辅助定位（避免框选偏置）
    st.sidebar.markdown("---")
    st.sidebar.markdown("**网格辅助**")
    show_grid = st.sidebar.checkbox("显示网格线", value=True, key="anno_show_grid",
        help="在图片上显示网格线，便于框选时准确定位目标物")
    grid_size = st.sidebar.number_input("网格间距(px)", min_value=10, max_value=200, value=50, step=10, key="anno_grid_size",
        help="网格线间距，像素越小网格越密") if show_grid else 50

    # 用于 canvas 的背景图：先缩放到显示尺寸，再叠加网格（确保网格与画布对齐）
    disp_img = img.resize((disp_w, disp_h), PILImage.LANCZOS) if (img.size != (disp_w, disp_h)) else img.copy()
    if show_grid and grid_size:
        disp_img = draw_grid_on_image(disp_img, grid_size=int(grid_size), color="#888888")

    # 模式：绘制椭圆 / 移动调整（移动时背景显示阴影作为相对位置参考）
    st.sidebar.markdown("**绘制模式**")
    canvas_mode = st.sidebar.radio(
        "模式",
        options=["circle", "transform"],
        format_func=lambda x: "绘制椭圆" if x == "circle" else "移动/调整（可拖拽修正位置）",
        index=0,
        key="anno_canvas_mode",
    )
    ann_objs = ann_data.get("objects", [])

    # 移动模式下：在背景上绘制阴影，作为相对移动参考
    disp_img_final = disp_img.copy()
    if canvas_mode == "transform" and ann_objs:
        disp_img_final = draw_shadow_objects_on_image(disp_img, ann_objs, disp_w, disp_h, color="#999999")

    # initial_drawing：使已有标注在 canvas 中显示并可移动（transform 模式）
    initial_drawing = None
    if ann_objs:
        _objs = []
        for o in ann_objs:
            ob = {
                "type": o.get("type", "ellipse"),
                "left": float(o.get("left", 0)),
                "top": float(o.get("top", 0)),
                "width": float(o.get("width", 0)),
                "height": float(o.get("height", 0)),
                "stroke": "#00FF00",
                "fill": "rgba(0, 255, 0, 0.2)",
                "scaleX": 1,
                "scaleY": 1,
            }
            _objs.append(ob)
        initial_drawing = {"objects": _objs, "version": "4.0.0"}

    if st_canvas is not None:
        col_canvas, col_preview = st.columns(2)
        with col_canvas:
            st.caption("📐 操作区（绘制 / 移动）")
            canvas_result = st_canvas(
                drawing_mode=canvas_mode,
                stroke_width=2,
                stroke_color="#00FF00",
                fill_color="rgba(0, 255, 0, 0.2)",
                background_image=disp_img_final,
                initial_drawing=initial_drawing,
                height=disp_h,
                width=disp_w,
                key="anno_canvas",
            )
        objs = []
        if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
            objs = canvas_result.json_data["objects"]
        ann_data["objects"] = [
            {k: o.get(k) for k in ("left", "top", "width", "height", "type") if k in o}
            for o in objs
        ]
        ann_data["canvas_width"] = disp_w
        ann_data["canvas_height"] = disp_h
        ann_data["image_width"] = img_w
        ann_data["image_height"] = img_h
        with col_preview:
            st.caption("📷 预览（保存后效果，与操作区同尺寸）")
            if objs:
                preview_base = disp_img.copy()
                preview_img = render_objects_on_image(preview_base, objs, disp_w, disp_h)
                st.image(preview_img, width=disp_w, channels="RGB",
                    caption="与左侧操作区同尺寸")
            else:
                st.image(disp_img, width=disp_w, channels="RGB",
                    caption="绘制框选后此处将显示预览")
        # 确认按钮：保存拉框标注（框选标记将永久留在图片上）
        if st.sidebar.button("确认 · 保存", type="primary", key="btn_confirm"):
            save_annotation(path, ann_data)
            st.session_state.annotation_saved_msg = "✅ 已保存"
            st.rerun()
    else:
        st.image(path, use_container_width=True)
        st.info("安装 streamlit-drawable-canvas-fix 可启用拉框标注: pip install streamlit-drawable-canvas-fix")


if not os.path.isdir(_DATA_ROOT):
    st.error(f"未找到数据根目录: {_DATA_ROOT}")
    st.info("请确保存在与 dataset_viewer 同级的 dataset 文件夹，且其下包含 run 文件夹（如 dataset0209），run 内有 episode_0/episode_1/.../episode_12，各 episode_N 下有 images 文件夹。")
    st.stop()

run_folders = get_run_folders(_DATA_ROOT)
if not run_folders:
    st.error(f"在 `{_DATA_ROOT}` 下未找到任何 run 子文件夹（如 dataset0209）。")
    st.stop()

# 若已选择进入标注工作台，则只渲染标注界面
if st.session_state.annotation_image_path:
    st.title("数据集标注平台")
    render_annotation_workspace()
    st.stop()

st.title("数据集标注平台")
st.caption(f"数据目录: `{_DATA_ROOT}` · **点击图片下方的「进入标注」可进入标注工作台**")

with st.sidebar:
    st.header("数据选择")
    selected_run = st.selectbox("Run 文件夹", run_folders, index=0)
    st.caption("平台优先显示 **images_png**（Zarr 导出后的 PNG），若无则显示 images。")
    st.divider()
    st.header("显示设置")
    grid_cols = st.slider("预览网格列数", 2, 8, 4)
    show_missing = st.checkbox("显示无数据的 ep", value=True)
    max_show = st.number_input("每个 ep 最多预览张数（0=全部）", 0, 5000, 48, step=24)
    st.divider()
    st.caption("若无法启动：请双击 **dataset_viewer/run.bat** 或在 CMD 中执行上述命令。")

ep_list = list_ep_folders(_DATA_ROOT, selected_run)
# 说明当前显示的图源（优先 images_png，即 Zarr 导出后的 PNG）
_source = IMAGES_PNG_SUBDIR if ep_list and ep_list[0][1] and (IMAGES_PNG_SUBDIR in (ep_list[0][1] or "")) else IMAGE_SUBFOLDER
st.caption(f"当前 Run: **{selected_run}** · 图源: **{_source}**（导出后的 PNG 在 `{IMAGES_PNG_SUBDIR}` 下） · 路径示例: `.../dataset/{selected_run}/episode_0/{_source}`")

# 已标注图片预览池
annotated_paths = []
for name, image_folder in ep_list:
    if not image_folder:
        continue
    _, paths = count_images(image_folder)
    for p in paths:
        if is_annotated(p):
            annotated_paths.append((name, p))
pool_cols = min(8, max(4, grid_cols + 2))
pool_max = 48  # 预览池最多展示张数
with st.expander(f"✅ 已标注图片预览池 · 共 **{len(annotated_paths)}** 张", expanded=(len(annotated_paths) > 0)):
    if not annotated_paths:
        st.caption("暂无已标注图片。点击下方各 ep 中的「进入标注」进行标注。")
    else:
        st.caption(f"展示前 {min(pool_max, len(annotated_paths))} 张 · 来源: ep0-ep12 · 删除后下方进度条将减一")
        for start in range(0, min(pool_max, len(annotated_paths)), pool_cols):
            row = annotated_paths[start : start + pool_cols]
            cols = st.columns(pool_cols)
            for c, (ep_n, img_path) in enumerate(row):
                with cols[c]:
                    try:
                        rendered = render_image_with_boxes(img_path)
                        disp_img = rendered if rendered is not None else img_path
                        if isinstance(disp_img, str):
                            st.image(disp_img, use_container_width=True)
                        else:
                            st.image(disp_img, use_container_width=True, channels="RGB")
                        st.caption(f"✅ 已标注 · **来源: {ep_n}** · {os.path.basename(img_path)}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("进入标注", key=f"pool_{ep_n}_{start}_{c}", use_container_width=True):
                                _, all_paths = count_images(os.path.dirname(img_path))
                                st.session_state.annotation_image_path = img_path
                                st.session_state.annotation_ep_name = ep_n
                                st.session_state.annotation_frame_name = os.path.basename(img_path)
                                st.session_state.annotation_image_paths = all_paths
                                st.session_state.annotation_image_index = all_paths.index(img_path)
                                st.session_state.annotation_saved_msg = None
                                st.rerun()
                        with col_b:
                            if st.button("取消标注", key=f"del_{ep_n}_{start}_{c}", use_container_width=True,
                                    help="移除本图标注，可重新标注"):
                                delete_annotation(img_path)
                                st.rerun()
                    except Exception as e:
                        st.caption(f"加载失败: {os.path.basename(img_path)}")

st.markdown("---")

for name, image_folder in ep_list:
    count, img_paths = count_images(image_folder) if image_folder else (0, [])
    if count == 0 and not show_missing:
        continue

    n_annotated = count_annotated(img_paths)
    prog = n_annotated / count if count else 0
    label = f"**{name}** · 共 **{count}** 张 · 已标注 **{n_annotated}/{count}**"
    with st.expander(label, expanded=(count > 0)):
        if count == 0:
            st.caption(f"该 ep 下暂无图片。请先运行 `zarr_to_png.py` 导出 PNG 到 **{IMAGES_PNG_SUBDIR}**，或确认 `episode_*/images` 内有图片。")
            continue
        st.progress(prog, text=f"已保存 {n_annotated}/{count} 张")
        st.caption(f"路径: `{image_folder}`")
        # 网格展示图片（支持限制数量，避免一次加载过多）
        paths_to_show = img_paths[:max_show] if max_show else img_paths
        if max_show and len(img_paths) > max_show:
            st.caption(f"共 {len(img_paths)} 张，仅展示前 {max_show} 张。")
        for start in range(0, len(paths_to_show), grid_cols):
            row_paths = paths_to_show[start : start + grid_cols]
            cols = st.columns(grid_cols)
            for c, img_path in enumerate(row_paths):
                with cols[c]:
                    try:
                        rendered = render_image_with_boxes(img_path) if is_annotated(img_path) else None
                        disp_img = rendered if rendered is not None else img_path
                        if isinstance(disp_img, str):
                            st.image(disp_img, use_container_width=True, caption=os.path.basename(img_path))
                        else:
                            st.image(disp_img, use_container_width=True, channels="RGB", caption=os.path.basename(img_path))
                        if is_annotated(img_path):
                            st.caption("✅ 已标注")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("进入标注", key=f"anno_{name}_{start}_{c}", use_container_width=True):
                                    st.session_state.annotation_image_path = img_path
                                    st.session_state.annotation_ep_name = name
                                    st.session_state.annotation_frame_name = os.path.basename(img_path)
                                    st.session_state.annotation_image_paths = img_paths
                                    st.session_state.annotation_image_index = img_paths.index(img_path)
                                    st.session_state.annotation_saved_msg = None
                                    st.rerun()
                            with col_b:
                                if st.button("取消标注", key=f"cancel_{name}_{start}_{c}", use_container_width=True,
                                        help="移除本图标注，可重新标注"):
                                    delete_annotation(img_path)
                                    st.rerun()
                        else:
                            if st.button("进入标注", key=f"anno_{name}_{start}_{c}", use_container_width=True):
                                st.session_state.annotation_image_path = img_path
                                st.session_state.annotation_ep_name = name
                                st.session_state.annotation_frame_name = os.path.basename(img_path)
                                st.session_state.annotation_image_paths = img_paths
                                st.session_state.annotation_image_index = img_paths.index(img_path)
                                st.session_state.annotation_saved_msg = None
                                st.rerun()
                    except Exception as e:
                        st.caption(f"加载失败: {os.path.basename(img_path)}")
                        st.code(str(e))

st.divider()
with st.expander("平台说明"):
    st.markdown("""
- **图源**：优先显示 **images_png**（由 `zarr_to_png.py` 从 Zarr chunk 导出），若无则显示 **images**。路径：`dataset/<run>/episode_0/images_png` 或 `.../images`。
- **标注工作台**：点击任意预览图下方的「进入标注」按钮，进入标注工作台，可在图片上画框、添加标签并保存（JSON 与图片同目录）。
- **已标注预览池**：展示当前 Run 下所有已保存标注的图片，可快速进入编辑。
- 支持格式：png, jpg, jpeg, bmp, webp, gif, tiff。
""")
    with st.expander("🔄 批量转 PNG（工具）", expanded=False):
        st.caption("将当前 Run 下所有 ep 的 images 中非 .png 转为 PNG（同主名 .png，删除原文件）。")
        if PILImage is None:
            st.warning("请先安装 Pillow: pip install Pillow")
        else:
            if st.button("执行转换（当前 Run 下全部 ep）", key="convert_png"):
                run_path = os.path.join(_DATA_ROOT, selected_run)
                total_ok = 0
                all_errors = []
                for name_ep, image_folder in ep_list:
                    if not image_folder:
                        continue
                    cnt, errs = convert_folder_to_png(image_folder)
                    total_ok += cnt
                    if errs:
                        all_errors.extend([f"[{name_ep}] {e}" for e in errs])
                st.success(f"转换完成，共 {total_ok} 张。")
                if all_errors:
                    st.code("\n".join(all_errors))
                st.rerun()
