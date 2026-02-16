"""
批量将 dataset 下各 run 的 episode_0..episode_12/images 中非 PNG 图片转为 PNG。
可被 app 或命令行脚本调用。
"""
import os

IMAGE_SUBFOLDER = "images"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif"}


def get_data_root():
    _app_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    return os.environ.get("DATA_ROOT") or os.path.normpath(os.path.join(_app_dir, "..", "dataset"))


def get_run_folders(root):
    if not os.path.isdir(root):
        return []
    return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]


def list_ep_image_folders(root, run_folder):
    result = []
    run_path = os.path.join(root, run_folder) if run_folder else root
    for i in range(13):
        episode_dir = os.path.join(run_path, f"episode_{i}")
        image_folder = os.path.join(episode_dir, IMAGE_SUBFOLDER)
        if os.path.isdir(image_folder):
            result.append((f"ep{i}", image_folder))
    return result


def convert_folder_to_png(folder_path):
    try:
        from PIL import Image
    except ImportError:
        return 0, ["Pillow 未安装: pip install Pillow"]
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
            img = Image.open(path).convert("RGB")
            img.save(png_path, "PNG")
            os.remove(path)
            converted += 1
        except Exception as e:
            errors.append(f"{f}: {e}")
    return converted, errors


def run_conversion_for_all_runs(data_root=None):
    """对 data_root 下所有 run 的 episode_0..12/images 执行转 PNG。返回 (总转换数, 错误列表)。"""
    root = data_root or get_data_root()
    if not os.path.isdir(root):
        return 0, [f"数据根目录不存在: {root}"]
    runs = get_run_folders(root)
    total = 0
    all_errors = []
    for run in runs:
        for name, image_folder in list_ep_image_folders(root, run):
            n, errs = convert_folder_to_png(image_folder)
            total += n
            if errs:
                all_errors.extend([f"[{run}/{name}] {e}" for e in errs])
    return total, all_errors


if __name__ == "__main__":
    import sys
    root = get_data_root()
    print(f"数据根目录: {root}")
    total, errors = run_conversion_for_all_runs(root)
    print(f"转换完成，共 {total} 张转为 PNG。")
    if errors:
        print("错误:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
