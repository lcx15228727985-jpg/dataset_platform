# 数据集标注平台 - 多阶段构建（前端 + 后端）
# 适用于 Linux 部署，单端口对外提供标注服务

# ---------- Stage 1: 构建前端 ----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci 2>/dev/null || npm install

COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: 运行后端（含静态文件） ----------
FROM python:3.11-slim
WORKDIR /app

# 系统依赖（Pillow 可选）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libwebp7 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt

COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist ./static

ENV PYTHONPATH=/app
ENV STATIC_DIR=/app/static
ENV DATA_ROOT=/data
ENV EXPORT_DIR=/data/export
EXPOSE 8000

# 默认 DATA_ROOT=/data，需通过 volume 挂载数据集
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
