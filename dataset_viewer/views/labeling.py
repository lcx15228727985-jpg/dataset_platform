"""
标注工作台：只改状态并调用 navigate_to('gallery')，不调用图库或 app。
禁止 import app。
"""
import base64
import io
import json
import os
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from utils import (
    draw_grid_on_image,
    draw_shadow_objects_on_image,
    load_annotation,
    render_objects_on_image,
    save_annotation,
    count_annotated,
)


def _save_confirm_dialog():
    """保存成功后的弹窗/内联：下一张 或 继续标注"""
    paths_list = st.session_state.get("_confirm_paths_list") or []
    idx = st.session_state.get("_confirm_idx", 0)
    n_total = len(paths_list)
    st.success("✅ 标注已保存。")
    st.caption("选择「下一张」进入下一张图标注，或「继续标注」留在当前图。")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("下一张", type="primary", key="btn_confirm_next", use_container_width=True):
            st.session_state.annotation_just_saved_confirm = False
            if idx < n_total - 1 and paths_list:
                st.session_state.annotation_image_path = paths_list[idx + 1]
                st.session_state.annotation_frame_name = os.path.basename(paths_list[idx + 1])
                st.session_state.annotation_image_index = idx + 1
            st.session_state.annotation_saved_msg = None
            st.rerun()
    with c2:
        if st.button("继续标注", key="btn_confirm_stay", use_container_width=True):
            st.session_state.annotation_just_saved_confirm = False
            st.session_state.annotation_saved_msg = None
            st.rerun()


_open_save_confirm_modal = None
if getattr(st, "dialog", None):
    @st.dialog("保存成功")
    def _open_save_confirm_modal():
        _save_confirm_dialog()


def _build_html_dual_canvas_config(disp_img_pil, disp_w, disp_h, initial_ellipse, show_grid, grid_size):
    """构建 HTML 双画布组件的 config JSON"""
    buf = io.BytesIO()
    disp_img_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    cfg = {
        "imageBase64": b64,
        "width": disp_w,
        "height": disp_h,
        "showGrid": bool(show_grid),
        "gridSize": int(grid_size) if show_grid else 50,
    }
    if initial_ellipse:
        o = initial_ellipse
        cfg["initialEllipse"] = {
            "left": float(o.get("left", 0)),
            "top": float(o.get("top", 0)),
            "width": float(o.get("width", 0)),
            "height": float(o.get("height", 0)),
            "scaleX": float(o.get("scaleX", 1)),
            "scaleY": float(o.get("scaleY", 1)),
            "originX": o.get("originX", "center"),
            "originY": o.get("originY", "center"),
        }
    return cfg


def render(navigate_to, app_dir):
    """标注工作台：只改状态 + navigate_to，不调用图库渲染。"""
    path = st.session_state.annotation_image_path
    ep_name = st.session_state.annotation_ep_name or ""
    paths_list = st.session_state.annotation_image_paths
    idx = st.session_state.annotation_image_index
    frame_name = st.session_state.annotation_frame_name or (os.path.basename(path) if path else "")

    if not path or not os.path.isfile(path):
        navigate_to("gallery")
        return

    # HTML 双画布保存回传：从 URL 读取 annot_data 并写入当前图片
    q = st.query_params
    if "annot_data" in q and path:
        try:
            raw = q["annot_data"]
            objs = json.loads(urllib.parse.unquote(raw))
            if isinstance(objs, list) and objs:
                ann_data = load_annotation(path)
                ann_data["objects"] = [
                    {k: o.get(k) for k in ("left", "top", "width", "height", "type", "scaleX", "scaleY", "originX", "originY") if k in o}
                    for o in objs
                ]
                ann_data["canvas_width"] = st.session_state.get("_html_disp_w") or 600
                ann_data["canvas_height"] = st.session_state.get("_html_disp_h") or 450
                ann_data["image_width"] = st.session_state.get("_html_img_w")
                ann_data["image_height"] = st.session_state.get("_html_img_h")
                if ann_data["image_width"] is None and PILImage:
                    try:
                        with PILImage.open(path) as im:
                            ann_data["image_width"], ann_data["image_height"] = im.size
                    except Exception:
                        pass
                save_annotation(path, ann_data)
                st.session_state.annotation_just_saved_confirm = True
                st.session_state._confirm_paths_list = st.session_state.annotation_image_paths
                st.session_state._confirm_idx = st.session_state.annotation_image_index
                new_params = {k: v for k, v in st.query_params.items() if k != "annot_data"}
                try:
                    if new_params:
                        st.query_params.from_dict(new_params)
                    else:
                        st.query_params.clear()
                except AttributeError:
                    st.query_params.clear()
                st.rerun()
        except Exception:
            pass

    st.subheader("📝 标注工作台")
    n_total = len(paths_list)
    n_annotated = count_annotated(paths_list)
    st.caption(f"**{ep_name}** · {frame_name} · 第 {idx + 1}/{n_total} 张 · 本 ep 已标注 {n_annotated}/{n_total} 张")

    prog_val = n_annotated / n_total if n_total else 0
    st.progress(prog_val, text=f"本 ep 标注进度：{n_annotated}/{n_total} 已保存")
    if st.session_state.annotation_saved_msg:
        st.success(st.session_state.annotation_saved_msg)
        st.session_state.annotation_saved_msg = None

    # 保存后弹窗/内联：仅渲染确认 UI，不渲染下方画布
    if st.session_state.get("annotation_just_saved_confirm"):
        if _open_save_confirm_modal is not None:
            _open_save_confirm_modal()
        else:
            st.info("✅ 标注已保存，请选择下一步：")
            _save_confirm_dialog()
        return

    col_ret, col_prev, col_next, _ = st.columns([1, 1, 1, 5])
    with col_ret:
        if st.button("← 返回图库", key="btn_ret"):
            navigate_to("gallery")
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

    if not PILImage:
        st.error("请安装 Pillow 以加载图片")
        return
    try:
        img = PILImage.open(path).convert("RGB")
    except Exception as e:
        st.error(f"无法加载图片: {e}")
        return

    img_w, img_h = img.size
    max_side = st.sidebar.slider(
        "显示最大边长(px)", min_value=400, max_value=1200, value=600, step=50,
        key="anno_max_side",
        help="操作区与预览区共用此值，缩小可改善双栏布局"
    )
    scale = min(1.0, max_side / max(img_w, img_h))
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**网格辅助**")
    show_grid = st.sidebar.checkbox("显示网格线", value=True, key="anno_show_grid",
        help="在图片上显示网格线，便于框选时准确定位目标物")
    grid_size = st.sidebar.number_input("网格间距(px)", min_value=10, max_value=200, value=50, step=10, key="anno_grid_size",
        help="网格线间距，像素越小网格越密") if show_grid else 50

    disp_img = img.resize((disp_w, disp_h), PILImage.LANCZOS) if (img.size != (disp_w, disp_h)) else img.copy()
    disp_img_with_grid = disp_img.copy()
    if show_grid and grid_size:
        disp_img_with_grid = draw_grid_on_image(disp_img_with_grid, grid_size=int(grid_size), color="#888888")

    ann_objs = ann_data.get("objects", [])
    _key_objs = "anno_canvas_objs_" + path.replace(os.sep, "_")
    _key_reset = _key_objs + "_reset"
    _prev_objs = st.session_state.get(_key_objs, [])
    _canvas_reset = st.session_state.get(_key_reset, 0)
    has_obj = bool(ann_objs or _prev_objs)
    canvas_mode = "transform" if has_obj else "circle"
    _initial_objs = ann_objs[:1] if ann_objs else _prev_objs[:1]

    disp_img_final = disp_img_with_grid.copy()
    if canvas_mode == "transform" and _initial_objs:
        disp_img_final = draw_shadow_objects_on_image(disp_img_with_grid, _initial_objs, disp_w, disp_h, color="#999999")

    initial_drawing = None
    if _initial_objs:
        _objs = []
        for o in _initial_objs:
            ob = {
                "type": o.get("type", "ellipse"),
                "left": float(o.get("left", 0)),
                "top": float(o.get("top", 0)),
                "width": float(o.get("width", 0)),
                "height": float(o.get("height", 0)),
                "stroke": "#00FF00",
                "fill": "rgba(0, 255, 0, 0.2)",
                "scaleX": float(o.get("scaleX", 1)),
                "scaleY": float(o.get("scaleY", 1)),
                "originX": o.get("originX", "center"),
                "originY": o.get("originY", "center"),
            }
            _objs.append(ob)
        initial_drawing = {"objects": _objs, "version": "4.0.0"}

    template_path = os.path.join(app_dir, "html_dual_canvas", "template.html")
    use_html_dual_canvas = os.path.isfile(template_path)
    if use_html_dual_canvas:
        st.session_state["_html_disp_w"] = disp_w
        st.session_state["_html_disp_h"] = disp_h
        st.session_state["_html_img_w"] = img_w
        st.session_state["_html_img_h"] = img_h
        cfg = _build_html_dual_canvas_config(
            disp_img,
            disp_w,
            disp_h,
            _initial_objs[0] if _initial_objs else None,
            show_grid,
            int(grid_size) if show_grid else 50,
        )
        config_json = json.dumps(cfg, ensure_ascii=False)
        config_escaped = urllib.parse.quote(config_json)
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read().replace("CONFIG_PLACEHOLDER", config_escaped)
        st.caption("📐 操作区（左） + 预览（右） · 同帧同步")
        components.html(html_content, height=disp_h + 220, scrolling=False)
    elif st_canvas is not None:
        col_canvas, col_preview = st.columns(2)
        with col_canvas:
            st.caption("📐 操作区（无标注时拖拽绘制椭圆，有标注时拖拽移动 · 每图仅一个框选）")
            canvas_result = st_canvas(
                drawing_mode=canvas_mode,
                stroke_width=2,
                stroke_color="#00FF00",
                fill_color="rgba(0, 255, 0, 0.2)",
                background_image=disp_img_final,
                initial_drawing=initial_drawing,
                height=disp_h,
                width=disp_w,
                key=f"anno_canvas_{_key_objs}_{_canvas_reset}",
            )
        objs = []
        if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
            objs = canvas_result.json_data["objects"]
        objs = objs[:1]
        st.session_state[_key_objs] = objs
        ann_data["objects"] = [
            {k: o.get(k) for k in ("left", "top", "width", "height", "type", "scaleX", "scaleY", "originX", "originY") if k in o}
            for o in objs
        ]
        ann_data["canvas_width"] = disp_w
        ann_data["canvas_height"] = disp_h
        ann_data["image_width"] = img_w
        ann_data["image_height"] = img_h
        with col_preview:
            st.caption("📷 预览（保存后效果，与操作区同尺寸）")
            if objs:
                preview_base = disp_img_final.copy()
                preview_img = render_objects_on_image(preview_base, objs, disp_w, disp_h)
                st.image(preview_img, width=disp_w, channels="RGB",
                    caption="与左侧操作区同尺寸")
            else:
                st.image(disp_img_with_grid, width=disp_w, channels="RGB",
                    caption="绘制框选后此处将显示预览")
        col_btn1, col_btn2 = st.sidebar.columns(2)
        with col_btn1:
            if st.button("确认 · 保存", type="primary", key="btn_confirm", use_container_width=True):
                save_annotation(path, ann_data)
                st.session_state.annotation_just_saved_confirm = True
                st.session_state._confirm_paths_list = paths_list
                st.session_state._confirm_idx = idx
                st.rerun()
        with col_btn2:
            if has_obj and st.button("清除重绘", key="btn_clear", use_container_width=True):
                ann_data["objects"] = []
                st.session_state[_key_objs] = []
                st.session_state[_key_reset] = _canvas_reset + 1
                save_annotation(path, ann_data)
                st.rerun()
    else:
        st.image(path, use_container_width=True)
        st.info("安装 streamlit-drawable-canvas-fix 可启用拉框标注: pip install streamlit-drawable-canvas-fix")
