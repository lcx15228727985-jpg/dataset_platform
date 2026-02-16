#!/bin/bash
# 数据集标注平台 - 后台启动（nohup），日志写入 streamlit.log
# 在项目根目录执行: bash run_background.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/dataset}"

if [ ! -d "venv" ]; then
    echo "未找到 venv，请先执行: bash install.sh"
    exit 1
fi
source venv/bin/activate

nohup streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
echo "已后台启动，PID: $!"
echo "日志: $SCRIPT_DIR/streamlit.log"
echo "访问: http://本机IP:8501"
