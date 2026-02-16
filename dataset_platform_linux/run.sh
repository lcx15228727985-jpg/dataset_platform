#!/bin/bash
# 数据集标注平台 - 前台启动（监听 0.0.0.0:8501）
# 在项目根目录执行: bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 数据根目录：与 dataset_viewer 同级的 dataset
export DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/dataset}"

if [ ! -d "venv" ]; then
    echo "未找到 venv，请先执行: bash install.sh"
    exit 1
fi
source venv/bin/activate

echo "数据目录: $DATA_ROOT"
echo "访问: http://本机IP:8501"
echo ""

exec streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501
