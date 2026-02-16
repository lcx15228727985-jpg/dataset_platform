#!/bin/bash
# 数据集标注平台 - 使用 Conda 环境时的前台启动
# 使用前请先创建并激活 conda 环境并安装依赖（见 部署说明.txt 方式 B）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/dataset}"

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "请先激活 conda 环境，例如: conda activate dataset_platform"
    exit 1
fi

echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "数据目录: $DATA_ROOT"
echo "访问: http://本机IP:8501"
echo ""

exec streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501
