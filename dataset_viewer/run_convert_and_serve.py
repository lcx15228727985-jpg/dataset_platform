"""
先自动执行「全部 Run 的 ep 转 PNG」，再启动可视化平台。
用法：在项目根目录执行  python dataset_viewer/run_convert_and_serve.py
"""
import os
import subprocess
import sys

# 确保从项目根运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import convert_to_png
get_data_root = convert_to_png.get_data_root
run_conversion_for_all_runs = convert_to_png.run_conversion_for_all_runs

def main():
    root = get_data_root()
    print(f"数据根目录: {root}")
    if not os.path.isdir(root):
        print("数据根目录不存在，跳过转换。")
    else:
        print("正在将各 ep 的 images 转为 PNG ...")
        total, errors = run_conversion_for_all_runs(root)
        print(f"转换完成，共 {total} 张转为 PNG。")
        if errors:
            for e in errors:
                print("  ", e)
    print("启动可视化平台 ...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(_SCRIPT_DIR, "app.py"),
        "--server.headless", "true",
    ], cwd=_PROJECT_ROOT)

if __name__ == "__main__":
    main()
