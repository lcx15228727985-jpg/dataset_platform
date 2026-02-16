#!/bin/bash
# 数据集标注平台 - Linux 依赖安装脚本
# 在项目根目录执行: bash install.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 数据集标注平台 - 安装 ==="
echo "项目根目录: $SCRIPT_DIR"

if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi
python3 --version

echo ""
echo "创建虚拟环境 venv ..."
python3 -m venv venv
source venv/bin/activate

echo "安装依赖 (dataset_viewer/requirements.txt) ..."
pip install -q -r dataset_viewer/requirements.txt

echo ""
echo "=== 安装完成 ==="
echo "运行服务: bash run.sh"
echo "或: source venv/bin/activate && streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501"
