"""
图库页：仅改状态并调用 navigate_to('workspace')，不调用标注页或 app。
"""
import os
import streamlit as st

from utils import (
    IMAGES_PNG_SUBDIR,
    IMAGE_SUBFOLDER,
    clear_canvas_state_for_path,
    convert_folder_to_png,
    count_annotated,
    count_images,
    delete_annotation,
    get_run_folders,
    is_annotated,
    list_ep_folders,
    render_image_with_boxes,
)


def render(navigate_to, data_root):
    run_folders = get_run_folders(data_root)
    st.title("数据集标注平台")
    st.caption(f"数据目录: `{data_root}` · **点击图片下方的「进入标注」可进入标注工作台**")

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

    ep_list = list_ep_folders(data_root, selected_run)
    _source = IMAGES_PNG_SUBDIR if ep_list and ep_list[0][1] and (IMAGES_PNG_SUBDIR in (ep_list[0][1] or "")) else IMAGE_SUBFOLDER
    st.caption(f"当前 Run: **{selected_run}** · 图源: **{_source}** · 路径示例: `.../dataset/{selected_run}/episode_0/{_source}`")

    _use_fragment = getattr(st, "fragment", None)

    def _pool_fragment_deco(f):
        return (_use_fragment(f) if _use_fragment else f)

    @_pool_fragment_deco
    def render_annotated_pool():
        annotated_paths = []
        for name, image_folder in ep_list:
            if not image_folder:
                continue
            _, paths = count_images(image_folder)
            for p in paths:
                if is_annotated(p):
                    annotated_paths.append((name, p))
        with st.expander(f"✅ 已标注图片预览池 · 共 **{len(annotated_paths)}** 张", expanded=(len(annotated_paths) > 0)):
            if not annotated_paths:
                st.caption("暂无已标注图片。点击下方各 ep 中的「进入标注」进行标注。")
            else:
                pool_cols = min(8, max(4, grid_cols + 2))
                pool_max = 48
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
                                        navigate_to("workspace")
                                with col_b:
                                    if st.button("取消标注", key=f"del_{ep_n}_{start}_{c}", use_container_width=True,
                                            help="移除本图标注，可重新标注"):
                                        delete_annotation(img_path)
                                        clear_canvas_state_for_path(img_path)
                                        if not _use_fragment:
                                            st.rerun()
                            except Exception as e:
                                st.caption(f"加载失败: {os.path.basename(img_path)}")

    render_annotated_pool()

    st.markdown("---")

    def _gallery_fragment_deco(f):
        return (_use_fragment(f) if _use_fragment else f)

    @_gallery_fragment_deco
    def render_ep_gallery():
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
                                            navigate_to("workspace")
                                    with col_b:
                                        if st.button("取消标注", key=f"cancel_{name}_{start}_{c}", use_container_width=True,
                                                help="移除本图标注，可重新标注"):
                                            delete_annotation(img_path)
                                            clear_canvas_state_for_path(img_path)
                                            if not _use_fragment:
                                                st.rerun()
                                else:
                                    if st.button("进入标注", key=f"anno_{name}_{start}_{c}", use_container_width=True):
                                        st.session_state.annotation_image_path = img_path
                                        st.session_state.annotation_ep_name = name
                                        st.session_state.annotation_frame_name = os.path.basename(img_path)
                                        st.session_state.annotation_image_paths = img_paths
                                        st.session_state.annotation_image_index = img_paths.index(img_path)
                                        st.session_state.annotation_saved_msg = None
                                        navigate_to("workspace")
                            except Exception as e:
                                st.caption(f"加载失败: {os.path.basename(img_path)}")
                                st.code(str(e))

    render_ep_gallery()

    st.divider()
    with st.expander("平台说明"):
        st.markdown("""
- **图源**：优先显示 **images_png**（由 `zarr_to_png.py` 从 Zarr chunk 导出），若无则显示 **images**。
- **标注工作台**：点击任意预览图下方的「进入标注」按钮，进入标注工作台，可在图片上画框、添加标签并保存（JSON 与图片同目录）。
- **已标注预览池**：展示当前 Run 下所有已保存标注的图片，可快速进入编辑。
- 支持格式：png, jpg, jpeg, bmp, webp, gif, tiff。
""")
        with st.expander("🔄 批量转 PNG（工具）", expanded=False):
            st.caption("将当前 Run 下所有 ep 的 images 中非 .png 转为 PNG（同主名 .png，删除原文件）。")
            try:
                from PIL import Image as PILImage
            except ImportError:
                PILImage = None
            if PILImage is None:
                st.warning("请先安装 Pillow: pip install Pillow")
            else:
                if st.button("执行转换（当前 Run 下全部 ep）", key="convert_png"):
                    run_path = os.path.join(data_root, selected_run)
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
