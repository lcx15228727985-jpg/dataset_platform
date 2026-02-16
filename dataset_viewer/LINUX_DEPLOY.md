# Linux 部署说明

适用于在 Linux 服务器上部署「数据集标注平台」，供多人协作标注，导出数据供后续训练使用。

---

## 一、部署包准备

### 方式 A：Git 拉取（推荐）

**在 Windows / Cursor 中：**

1. 项目根目录已包含 `.gitignore`（排除 node_modules、__pycache__、dist 等）
2. 初始化并提交到 Git 仓库：
   ```bash
   git init
   git add .
   git commit -m "feat: 数据集标注平台 Linux 部署"
   git remote add origin <你的仓库地址>
   git push -u origin main
   ```

**在 Linux / Cursor 中：**

1. 拉取代码：
   ```bash
   git clone <你的仓库地址>
   cd IROS
   cd dataset_viewer
   ```
2. 按「三、Docker 部署」继续

### 方式 B：ZIP 打包

1. 运行 `pack_for_linux.bat` 生成 `dataset_viewer_linux.zip`
2. 传到 Linux 后解压，进入 `dataset_viewer` 目录

---

## 二、目录结构要求

部署后目录结构示例：

```
dataset_viewer/           # 项目根
├── Dockerfile
├── docker-compose.yml
├── backend/
├── frontend/
└── ...

/data                     # 数据集根目录（可挂载到其他路径）
├── run1/
│   ├── episode_0/
│   │   └── images/       # 或 images_png/
│   ├── episode_1/
│   └── ...
├── run2/
└── export/               # 导出目录（标注 JSON/CSV）
```

- **DATA_ROOT**：数据集根目录，包含 `run/episode/images` 结构
- **EXPORT_DIR**：导出目录，默认 `DATA_ROOT/export`，导出的标注 JSON/CSV 会保存在此

---

## 三、Docker 部署（推荐）

### 3.1 安装 Docker 与 Docker Compose

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose

# 或使用 Docker 官方脚本
curl -fsSL https://get.docker.com | sh
```

### 3.2 准备数据目录

```bash
# 创建数据集根目录（若尚未存在）
sudo mkdir -p /data
sudo chown $USER:$USER /data

# 确保 run/episode/images 结构已放入 /data
# 例如: /data/dataset0209/episode_0/images/
```

### 3.3 构建并启动

```bash
cd dataset_viewer

# 使用默认数据目录 ./data（需提前创建或使用环境变量）
export DATA_ROOT=/data
docker-compose up -d --build

# 或直接指定数据路径
DATA_ROOT=/path/to/your/dataset docker-compose up -d --build
```

### 3.4 初始化数据库

首次部署需扫描数据集并建库：

```bash
# 进入容器执行
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.init_pg_db 2>/dev/null || python -m backend.scan_dataset"
```

- 若已配置 `DATABASE_URL`（PostgreSQL），会使用 `init_pg_db`
- 否则使用 SQLite：`scan_dataset`

### 3.5 验证

- 打开浏览器访问：`http://<Linux服务器IP>:8000`
- API 健康检查：`http://<Linux服务器IP>:8000/api/health`
- 若返回 `{"ok":true,...}` 即表示部署成功

---

## 四、非 Docker 部署（可选）

若不便使用 Docker，可手动安装环境：

### 4.1 安装 Node.js 与 Python

```bash
# Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip
```

### 4.2 构建前端

```bash
cd dataset_viewer/frontend
npm install
npm run build
```

### 4.3 运行后端

```bash
cd dataset_viewer
export DATA_ROOT=/data
export EXPORT_DIR=/data/export
export STATIC_DIR=$(pwd)/frontend/dist

python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

python -m backend.init_pg_db 2>/dev/null || python -m backend.scan_dataset
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 五、环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| DATA_ROOT | 数据集根目录 | `../dataset` 或容器内 `/data` |
| EXPORT_DIR | 导出标注保存目录 | `DATA_ROOT/export` |
| STATIC_DIR | 前端静态文件目录（Docker 自动设置） | - |
| DATABASE_URL | PostgreSQL 连接串（可选） | - |

---

## 六、导出与训练

1. 在平台中点击「保存到服务器 (JSON)」或「导出标注 (JSON)」
2. 导出的 `annotations.json` 会保存到 `EXPORT_DIR`（默认 `/data/export`）
3. 训练脚本可从 `EXPORT_DIR` 读取标注，结合 `DATA_ROOT` 下的图片进行训练

---

## 七、常见问题

### 7.1 端口 8000 被占用

修改 `docker-compose.yml` 中的端口映射，例如：`"9000:8000"`

### 7.2 数据集路径权限

确保容器内可读：`chmod -R 755 /data` 或挂载时使用正确用户

### 7.3 导出 404

- 确认已执行 `init_pg_db` 或 `scan_dataset`，数据库中有数据
- 确认后端健康：`curl http://localhost:8000/api/health`
