"""
后端数据层：与 dataset_viewer 同结构的 run/episode/images 与标注 JSON。
无 Streamlit 依赖，供 FastAPI 使用。
"""
import json
import os

IMAGE_SUBFOLDER = "images"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}
IMAGES_PNG_SUBDIR = "images_png"


def get_data_root():
    """数据根目录：环境变量 DATA_ROOT，或 与 dataset_viewer 同级的 dataset（backend 在 dataset_viewer/backend 下）"""
    _dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    # backend/ -> dataset_viewer/ -> 项目根，dataset 在项目根下
    return os.environ.get("DATA_ROOT") or os.path.normpath(os.path.join(_dir, "..", "..", "dataset"))


def get_run_folders(root):
    if not root or not os.path.isdir(root):
        return []
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def list_ep_folders(root, run_folder):
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
    if not folder_path or not os.path.isdir(folder_path):
        return 0, []
    paths = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXT:
            paths.append(os.path.join(folder_path, f))
    paths.sort(key=lambda p: os.path.basename(p))
    return len(paths), paths


def count_images_on_disk(data_root: str) -> dict:
    """与 init_pg_db/scan_and_fill 相同的扫描逻辑，仅统计不写库。用于部署校验。"""
    if not data_root or not os.path.isdir(data_root):
        return {"runs": 0, "episodes": 0, "total_images": 0}
    runs = get_run_folders(data_root)
    total_images = 0
    total_episodes_with_images = 0
    for run_name in runs:
        ep_list = list_ep_folders(data_root, run_name)
        for _ep_name, image_folder in ep_list:
            if not image_folder or not os.path.isdir(image_folder):
                continue
            n, _ = count_images(image_folder)
            if n > 0:
                total_episodes_with_images += 1
            total_images += n
    return {
        "runs": len(runs),
        "episodes": total_episodes_with_images,
        "total_images": total_images,
    }


def get_annotation_path(image_path):
    return os.path.splitext(image_path)[0] + "_annot.json"


def is_annotated(image_path):
    return os.path.isfile(get_annotation_path(image_path))


def count_annotated(paths):
    return sum(1 for p in paths if is_annotated(p))


def load_annotation(image_path):
    p = get_annotation_path(image_path)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"labels": [], "objects": []}


def save_annotation(image_path, data):
    p = get_annotation_path(image_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_annotation(image_path):
    p = get_annotation_path(image_path)
    if os.path.isfile(p):
        os.remove(p)


def image_path_to_id(root, image_path):
    """将绝对路径转为相对 data root 的 id（URL 用正斜杠）"""
    rel = os.path.relpath(image_path, root)
    return rel.replace("\\", "/")


def image_id_to_path(root, image_id):
    """将 id 转为绝对路径，并校验不越界"""
    if ".." in image_id or image_id.startswith("/"):
        return None
    path = os.path.normpath(os.path.join(root, *image_id.split("/")))
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    if not path_real.startswith(root_real + os.sep) and path_real != root_real:
        return None
    return path
