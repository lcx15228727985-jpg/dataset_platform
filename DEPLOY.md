# Linux 服务器部署说明（Docker + 数据库）

## 1. 环境要求

- Docker 与 Docker Compose
- 数据集目录：宿主机上已有 `run/episode_*/images`（或 `images_png`）结构

## 2. 启动数据库与后端

```bash
# 在项目根目录（含 docker-compose.yml）
docker compose up -d

# 仅启动数据库（本机跑后端时）
docker compose up -d db
```

- **db**：PostgreSQL 15，端口 5432，数据持久化在 volume `pg_data`
- **backend**：FastAPI，端口 8000，环境变量 `DATABASE_URL` 指向 `db`，`DATA_ROOT=/data/dataset`

## 3. 挂载数据集到后端

默认 compose 中 `backend` 使用匿名卷 `dataset_volume`。生产环境应改为挂载宿主机目录：

在 `docker-compose.yml` 中修改 backend 的 volumes：

```yaml
volumes:
  - /path/on/host/dataset:/data/dataset   # 宿主机实际数据集路径
```

然后重启：`docker compose up -d --build`

## 4. 初始化数据库（必做一次）

将 run/episode 下的图片元数据写入 Postgres，后续接口才会有 run 列表与分页。

**方式 A：在宿主机执行（推荐）**

宿主机已安装 Python 与依赖，且能访问数据库端口时：

```bash
cd dataset_viewer
export DATABASE_URL=postgresql://admin:password123@localhost:5432/annotation_system
export DATA_ROOT=/path/to/your/dataset   # 与 backend 容器内看到的路径一致；若只在本机 init，用本机路径
pip install -r backend/requirements.txt
python -m backend.init_pg_db
```

**方式 B：在 backend 容器内执行**

数据集已挂载到容器内 `/data/dataset` 时：

```bash
docker compose exec backend python -m backend.init_pg_db
```

（容器内 `DATA_ROOT` 已为 `/data/dataset`，无需再设）

## 5. 前端构建与访问

前端需单独构建，并指定后端 API 地址（若前后端不同源）：

```bash
cd dataset_viewer/frontend
# 若前端与后端同机部署且通过 Nginx 反代到同一域名，可留空
export VITE_API_URL=http://your-server:8000
npm install && npm run build
```

将 `dist/` 部署到 Nginx 或任意静态服务器；若与后端同域且 `/api` 反代到后端，则 `VITE_API_URL` 可为空。

## 6. 健康检查与入库校验

- **健康检查**：`GET http://localhost:8000/api/health`  
  返回 `database`、`data_root`、`data_root_exists` 等，便于排查。
- **部署/入库情况**：`GET http://localhost:8000/api/deploy/status`  
  - 返回库内统计：`db.runs`、`db.episodes`、`db.images`。
  - 加参数 **`?verify=1`** 会同时按与 init 相同的逻辑扫描 `DATA_ROOT` 磁盘，对比库内图片数与磁盘图片数：
    - `consistent: true` 表示图片信息已全部入库；
    - `consistent: false` 时可根据 `db.images` 与 `disk.total_images` 判断是否需重新执行 `init_pg_db` 或 `scan_dataset`。
- 数据库：compose 中 `db` 已配置 `pg_isready` 健康检查。

## 7. 仅数据库（本机开发）

只起 Postgres，后端在本机跑：

```bash
docker compose up -d db
# 本机
cd dataset_viewer
set DATABASE_URL=postgresql://admin:password123@localhost:5432/annotation_system
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

本机执行 `python -m backend.init_pg_db` 时，`DATA_ROOT` 设为本地数据集路径即可。
