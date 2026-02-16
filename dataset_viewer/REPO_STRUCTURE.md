# 仓库结构说明（精简版 / 阿里云部署用）

提交到 Git 并在阿里云拉取部署时，可参考本说明区分**必需**与**可选/冗余**文件。

---

## 一、Docker 部署必需（阿里云拉取后必留）

| 路径 | 说明 |
|------|------|
| `Dockerfile` | 镜像构建 |
| `docker-compose.yml` | 编排与挂载 |
| `.dockerignore` | 构建排除 |
| `backend/` | 后端全部（main.py、data.py、db.py、pg_db.py、scan_dataset.py、init_pg_db.py、requirements.txt） |
| `frontend/` | 前端源码（package.json、src/ 等，不含 node_modules） |
| `ALIYUN_LINUX_DEPLOY.md` | 阿里云部署步骤 |
| `LINUX_DEPLOY.md` | 通用 Linux 部署说明 |
| `ACCESS_GUIDE.md` | 多设备访问说明 |

---

## 二、批处理文件（仅 Windows 使用，Linux 可忽略）

| 文件 | 用途 | Linux 上 |
|------|------|----------|
| `run_all_local.bat` | Windows 本地一键启动后端+前端 | 不用，保留在仓库无影响 |
| `run_remove_images_keep_png.bat` | Windows 下执行「删 images 保留 images_png」 | 在 Linux 用：`python remove_images_keep_png.py /data` |
| ~~pack_for_linux.bat~~ | 已删除（已改用 Git 拉取） | - |

---

## 三、脚本与文档（按需保留）

| 路径 | 说明 |
|------|------|
| `remove_images_keep_png.py` | 删除 episode 下 images 仅保留 images_png，建议保留 |
| `zarr_to_png.py` | Zarr 转 PNG，有 Zarr 数据时有用 |
| `README_BS.md` | B/S 架构说明，已改为引用 run_all_local.bat |
| ~~DEPLOY.md~~ | 已删除（Streamlit 版文档，与当前 Docker 部署无关） |
| ~~convert_to_png.py~~, ~~convert_images_to_png.py~~, ~~run_convert_and_serve.py~~ | 已删除（图片转换脚本，非部署必需） |

---

## 四、Streamlit 相关（可选，不影响 Docker 部署）

以下为旧版 Streamlit 入口与视图，当前阿里云部署**仅用 FastAPI + React**，不依赖这些文件。若要做「极简仓库」可整块删除：

- `app.py`（Streamlit 入口）
- `views/`（home.py, labeling.py, __init__.py）
- `utils.py`（若仅被 Streamlit 引用）
- `dataset_viewer/requirements.txt`（Streamlit 依赖，与 `backend/requirements.txt` 不同）

删除后需确认无其他代码引用上述模块。

---

## 五、建议提交前自检

1. **已修复**：`README_BS.md`、`views/home.py` 中已不再引用已删除的 `run_bs_backend.bat`、`run_bs_frontend.bat`、`run.bat`，统一为 `run_all_local.bat`。
2. **无失效批处理**：当前 3 个 .bat 均在 Windows 下可用；Linux 部署不执行任何 .bat。
3. **已删以精简**：`pack_for_linux.bat`、`DEPLOY.md`、`convert_to_png.py`、`convert_images_to_png.py`、`run_convert_and_serve.py`。若不需要 Streamlit，可再删 `app.py`、`views/`、`utils.py`、根目录 `requirements.txt`。

按上述整理后即可提交精简版到 Git，并在阿里云上拉取后按 `ALIYUN_LINUX_DEPLOY.md` 部署。
