# 数据集标注平台 · B/S 架构

前后端分离：**FastAPI 后端** 只负责提供图片与保存标注，**React + Konva 前端** 负责全部界面与画布交互，零服务端渲染延迟。

## 技术栈

- **后端**: FastAPI + uvicorn（Python）
- **前端**: React 18 + Vite + react-konva（Canvas 双画布同步）

## 45k+ 图量加速（可选）

当单 run 下图片数量很大（如 45,000 张）时，可启用「数据库 + 游标分页 + 虚拟列表」：

1. **一次性建库**（在 dataset_viewer 目录下）：
   ```bash
   python -m backend.scan_dataset
   ```
   会在 `DATA_ROOT` 下生成 `catalog.db`，将 run/episode/图片路径与标注状态写入 SQLite。

2. **后端行为**：若存在 `catalog.db`，则
   - `/api/runs`、`/api/runs/{run}/episodes` 从数据库读（episodes 不再带完整 images 列表），并返回 `cursorAvailable: true`
   - 新增游标分页：`GET /api/runs/{run}/episodes/{ep}/images?cursor=&limit=50`，返回 `{ items, nextCursor }`

3. **前端行为**：当 `cursorAvailable === true` 时，
   - 图库按 episode 用 **react-virtuoso** 虚拟列表 + 游标分页加载图片，只渲染可见行
   - 列表缩略图请求带 `thumb=1`，后端按需生成约 150px 缩略图，减少传输

4. **未建库时**：与之前一致，episodes 仍返回完整 images 列表，无游标接口。

## 目录结构

```
dataset_viewer/
├── backend/
│   ├── main.py         # FastAPI 入口与 API
│   ├── data.py         # 数据路径与标注读写（无 Streamlit）
│   ├── db.py            # SQLite 元数据与游标分页（45k 加速）
│   ├── scan_dataset.py  # 一次性扫描建库脚本
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/client.js   # 请求后端 API
│   │   ├── pages/
│   │   │   ├── Gallery.jsx       # 图库（Run/Ep 选择、进入标注）
│   │   │   └── Annotation.jsx    # 双画布标注工作台
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── run_all_local.bat   # Windows 一键启动前后端（本地开发）
└── README_BS.md
```

## 启动方式

### 1. 后端（必须先启动）

在 **dataset_viewer** 目录下：

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

或使用 **run_all_local.bat**（Windows 一键启动后端+前端）。

- API 文档: http://localhost:8000/docs
- 数据根目录: 环境变量 `DATA_ROOT`，或默认 `项目根/dataset`（与 dataset_viewer 同级）

### 2. 前端

在 **dataset_viewer** 目录下：

```bash
cd frontend
npm install
npm run dev
```

或使用 **run_all_local.bat**（Windows 一键启动后端+前端）。

- 页面: http://localhost:5173
- 开发时 Vite 会把 `/api` 代理到 `http://localhost:8000`，无需配置 CORS。

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/runs | Run 文件夹列表 |
| GET | /api/runs/{run}/episodes | 该 Run 下各 episode 及图片列表（含 id、filename、annotated） |
| GET | /api/image/{image_id} | 图片文件（image_id 为相对 data root 的路径，如 `run0/episode_0/images_png/frame.png`） |
| GET | /api/annotation/{image_id} | 标注 JSON |
| POST | /api/annotation/{image_id} | 保存标注 |
| DELETE | /api/annotation/{image_id} | 删除标注 |

## 前端行为

- **图库**: 选择 Run → 展示各 ep 图片网格，点击「进入标注」跳转标注页并传入当前图及同 ep 的 images 列表（用于上一张/下一张）。
- **标注页**: 左画布操作（拖拽绘制椭圆、移动/缩放已有椭圆）、右画布同状态预览；鼠标十字线仅左侧显示；保存/清除/上一张/下一张/返回图库。
- 标注格式与原有 Streamlit 版一致（同目录 `_annot.json`，Fabric 风格 objects），可与现有数据兼容。

## 与 Streamlit 版对比

| 项目 | Streamlit 版 | B/S 版 |
|------|-------------|--------|
| 页面嵌套 | 需中央调度器避免套娃 | SPA 单页，无嵌套 |
| 画布同步 | 依赖 iframe/组件回传 | 前端同一 state，双 Stage 同帧 |
| 延迟 | 操作需回传服务端再渲染 | 纯前端渲染，&lt;16ms |
| 扩展 | 受限于 Streamlit 生态 | 可加撤销、多人、精细交互 |

原 Streamlit 入口仍保留在 `app.py`，可按需选用。
