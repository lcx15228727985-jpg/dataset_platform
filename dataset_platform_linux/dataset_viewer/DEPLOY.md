# 曲率数据集可视化平台 · 服务器部署资源申请文档

## 一、应用概述

**应用名称**：曲率数据集可视化平台  
**技术栈**：Python 3.8+ / Streamlit / Pillow  
**数据存储**：基于本地文件系统（图片 + JSON 标注），无数据库依赖  

**核心功能**：
- 按 Run（如 dataset0209）浏览 ep0 ~ ep12 图片
- 图片标注工作台：画框、保存、修改
- 已标注图片预览池：展示来源、支持删除
- 图片增删改查：浏览、新增标注、修改标注、删除标注，进度统计

---

## 二、功能与资源对应关系

| 功能         | 说明                         | 资源影响     |
|--------------|------------------------------|--------------|
| **查**       | 浏览、预览 ep0-ep12 图片     | 内存、磁盘 I/O |
| **增**       | 进入标注工作台画框并保存     | CPU、磁盘写 |
| **改**       | 重新打开标注、修改框并保存   | 同上         |
| **删**       | 删除预览池中的标注（保留原图）| 磁盘写       |
| **批量展示** | 网格展示多张图片、带框渲染   | 内存、CPU    |

---

## 三、服务器配置要求

### 3.1 推荐配置（中小规模数据集）

| 项目     | 要求        | 说明                                      |
|----------|-------------|-------------------------------------------|
| **CPU**  | 2 核及以上  | 单进程 Streamlit，主要处理图片加载与渲染  |
| **内存** | 4 GB 及以上 | 单用户约 0.5–1 GB，多用户需预留余量       |
| **磁盘** | 50 GB 及以上| 存放数据集（图片 + 标注 JSON），建议 SSD  |
| **系统** | Linux x64   | Ubuntu 20.04 / 22.04 或 CentOS 7+         |

### 3.2 轻量配置（试运行 / 小数据集）

| 项目     | 要求        |
|----------|-------------|
| CPU      | 1 核        |
| 内存     | 2 GB        |
| 磁盘     | 20 GB       |

### 3.3 高并发 / 大数据集

| 项目     | 建议        |
|----------|-------------|
| CPU      | 4 核        |
| 内存     | 8 GB        |
| 磁盘     | 100 GB+ SSD |

---

## 四、软件环境要求

| 软件           | 版本          |
|----------------|---------------|
| Python         | 3.8+          |
| streamlit      | ≥ 1.28.0      |
| Pillow         | ≥ 10.0.0      |
| zarr           | ≥ 2.14.0（可选，用于 Zarr 导出） |
| numpy          | ≥ 1.24.0      |

---

## 五、磁盘与数据说明

- **数据根目录**：`dataset/`，可通过环境变量 `DATA_ROOT` 覆盖
- **目录结构示例**：`dataset/<run>/episode_<N>/images` 或 `images_png`
- **标注文件**：与图片同目录的 `*_annot.json`，与图片一一对应
- **存储特点**：纯文件读写，无数据库；支持图片浏览、标注新增、修改、删除

---

## 六、网络与访问

| 项目           | 说明                                  |
|----------------|---------------------------------------|
| 端口           | 8501（Streamlit 默认）                |
| 访问方式       | `http://<服务器IP>:8501`              |
| 内网 / 公网    | 视安全策略开放 8501 端口或通过反向代理 |

---

## 七、部署步骤简述

1. 安装 Python 3.8+ 及依赖（见 `requirements.txt`）
2. 将项目代码及 `dataset/` 数据拷贝至服务器
3. 设置 `DATA_ROOT`（可选）指向实际数据目录
4. 执行：`streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501`
5. 使用 `nohup` 或 systemd 等方式保持后台运行

---

## 八、资源申请摘要（可复制用于申请单）

```
应用：曲率数据集可视化平台
用途：图片浏览与标注（增删改查），支持 ep0-ep12 数据集管理
技术：Streamlit + Pillow，文件存储

申请配置：
  - CPU：2 核
  - 内存：4 GB
  - 磁盘：50 GB（建议 SSD）
  - 系统：Linux x64（Ubuntu 20.04 / 22.04 或 CentOS 7+）
  - 端口：8501（HTTP）
```

---

## 九、备注

- 若仅用于内网试运行，可采用 **轻量配置**（1 核 / 2 GB / 20 GB）
- 若需多人同时标注或数据集较大，建议采用 **推荐配置** 及以上
- 数据目录建议挂载持久化存储，保证标注增删改查结果不丢失

---

## 十、Linux 服务器配置目录（逐步说明）

将一台 Linux 电脑作为服务器时，按以下目录和步骤配置即可。

### 10.1 推荐目录结构

在服务器上规划一个**项目根目录**（例如 `/opt/dataset_platform` 或 `/home/你的用户名/dataset_platform`），结构如下：

```
/opt/dataset_platform/              # 项目根目录（可自定）
├── dataset_viewer/                 # 应用代码目录
│   ├── app.py
│   ├── requirements.txt
│   ├── DEPLOY.md
│   └── ...
├── dataset/                         # 数据根目录（必须与 dataset_viewer 同级，或后用 DATA_ROOT 指向）
│   ├── dataset0209/                 # 一个 run
│   │   ├── episode_0/
│   │   │   ├── images/
│   │   │   │   ├── frame_00000.png
│   │   │   │   └── ...
│   │   │   └── images_png/          # 可选，Zarr 导出后的 PNG
│   │   ├── episode_1/
│   │   └── ...
│   └── 其他run/
└── venv/                            # 可选：Python 虚拟环境
```

**要点**：
- `dataset_viewer/` 与 `dataset/` 放在**同一级**，应用默认会找 `../dataset` 作为数据根。
- 若数据放在别处（如 `/data/dataset`），则通过环境变量 `DATA_ROOT` 指定，见下。

### 10.2 逐步配置

**步骤 1：创建项目根目录**

```bash
sudo mkdir -p /opt/dataset_platform
sudo chown $USER:$USER /opt/dataset_platform
cd /opt/dataset_platform
```

（若用用户目录：`mkdir -p ~/dataset_platform && cd ~/dataset_platform`）

**步骤 2：上传代码**

- 将本地的 `dataset_viewer` 整个文件夹拷贝到服务器 `/opt/dataset_platform/dataset_viewer`。
- 可用 `scp`、`rsync` 或 Git 拉取，例如：

```bash
# 示例：用 scp 从本机上传（在本机执行）
scp -r D:\PyCharmProject\gpr_prj\IROS\dataset_viewer 用户@服务器IP:/opt/dataset_platform/
```

**步骤 3：创建数据目录**

```bash
mkdir -p /opt/dataset_platform/dataset
```

- 将已有 run（如 `dataset0209`）上传到 `dataset/` 下，保持 `dataset/<run>/episode_<N>/images` 或 `images_png` 结构。

**步骤 4：数据不在项目根时（可选）**

若数据在例如 `/data/dataset`，则**不**在项目根下建 `dataset`，改为之后用环境变量指定：

```bash
export DATA_ROOT=/data/dataset
```

（在 systemd 或启动脚本里写 `Environment="DATA_ROOT=/data/dataset"` 或 `export DATA_ROOT=...`）

**步骤 5：安装 Python 与依赖**

```bash
cd /opt/dataset_platform
python3 --version   # 需 3.8+

# 建议使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r dataset_viewer/requirements.txt
```

**步骤 6：前台试运行**

```bash
cd /opt/dataset_platform
source venv/bin/activate
export DATA_ROOT=/opt/dataset_platform/dataset   # 若数据在别处，改为实际路径
streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

- 本机访问：`http://localhost:8501`
- 同网段访问：`http://服务器IP:8501`

**步骤 7：开放防火墙端口（如需外网或内网访问）**

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 8501/tcp
sudo ufw reload

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

**步骤 8：后台常驻（二选一）**

- **方式 A：nohup**

```bash
cd /opt/dataset_platform
source venv/bin/activate
export DATA_ROOT=/opt/dataset_platform/dataset
nohup streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

- **方式 B：systemd（推荐，开机自启）**

创建服务文件：

```bash
sudo nano /etc/systemd/system/dataset-platform.service
```

写入（按实际路径修改）：

```ini
[Unit]
Description=数据集标注平台 Streamlit
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/opt/dataset_platform
Environment="PATH=/opt/dataset_platform/venv/bin"
Environment="DATA_ROOT=/opt/dataset_platform/dataset"
ExecStart=/opt/dataset_platform/venv/bin/streamlit run dataset_viewer/app.py --server.address 0.0.0.0 --server.port 8501
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

然后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable dataset-platform
sudo systemctl start dataset-platform
sudo systemctl status dataset-platform
```

### 10.3 配置目录小结

| 用途           | 目录或变量 |
|----------------|------------|
| 项目根         | 如 `/opt/dataset_platform` |
| 应用代码       | `项目根/dataset_viewer/` |
| 数据根（默认） | `项目根/dataset/` |
| 数据根（自定义） | 环境变量 `DATA_ROOT`（如 `/data/dataset`） |
| 虚拟环境       | 可选 `项目根/venv/` |
| 运行/日志      | systemd 或 nohup 的配置中指定工作目录为项目根 |
