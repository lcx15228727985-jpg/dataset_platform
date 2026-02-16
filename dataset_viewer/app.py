"""
曲率数据集可视化平台 · 中央调度器入口。

唯一入口：本文件根据 session state 决定渲染哪个页面；子页面只通过 navigate_to 切换，禁止 import app。
依赖方向：app → views → utils；views 不引用 app。
"""
import os

# 避免本机 SSL/OpenMP 导致无法启动
os.environ.pop("SSLKEYLOGFILE", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from utils import get_run_folders

# 必须最先调用
st.set_page_config(page_title="数据集标注平台", layout="wide")

# Session state 初始化
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
    st.session_state.annotation_saved_msg = None
if "annotation_just_saved_confirm" not in st.session_state:
    st.session_state.annotation_just_saved_confirm = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "gallery"

# 数据根目录（与 dataset_viewer 同级的 dataset，可由环境变量 DATA_ROOT 覆盖）
_APP_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.normpath(os.path.join(_APP_DIR, "..", "dataset"))

# 启动前校验数据目录
if not os.path.isdir(_DATA_ROOT):
    st.error(f"未找到数据根目录: {_DATA_ROOT}")
    st.info("请确保存在与 dataset_viewer 同级的 dataset 文件夹，且其下包含 run 文件夹（如 dataset0209）。")
    st.stop()

run_folders = get_run_folders(_DATA_ROOT)
if not run_folders:
    st.error(f"在 `{_DATA_ROOT}` 下未找到任何 run 子文件夹（如 dataset0209）。")
    st.stop()

# 主内容区单一容器，切换页面时先清空再绘制，避免嵌套
main_placeholder = st.empty()


def navigate_to(page_name):
    """
    唯一的页面切换入口（中央调度器模式）。
    子页面只可调用此函数，禁止直接调用其他页的 render 或 import app。
    """
    st.session_state.current_page = page_name
    if page_name == "gallery":
        st.session_state.annotation_image_path = None
        st.session_state.annotation_ep_name = None
        st.session_state.annotation_frame_name = None
        st.session_state.annotation_image_paths = []
        st.session_state.annotation_image_index = 0
        st.session_state.annotation_saved_msg = None
    st.rerun()


def main():
    """中央调度：根据状态只渲染一个页面；子页面通过 navigate_to 切换，绝不互相调用。"""
    st.sidebar.caption("数据集标注平台 · 全局导航")

    if st.session_state.current_page == "workspace" and st.session_state.annotation_image_path:
        from views import labeling
        with main_placeholder.container():
            st.title("数据集标注平台")
            labeling.render(navigate_to, app_dir=_APP_DIR)
        return

    st.session_state.current_page = "gallery"
    from views import home
    with main_placeholder.container():
        home.render(navigate_to, data_root=_DATA_ROOT)


# 唯一入口
main()
