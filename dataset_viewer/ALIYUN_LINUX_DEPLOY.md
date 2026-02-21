# 阿里云 Linux 服务器部署标注平台

本文档给出在**阿里云 ECS（Alibaba Cloud Linux）**上从零部署「数据集标注平台」的完整命令，按顺序执行即可。

---

## 一、部署前准备（阿里云控制台）

1. **ECS 实例**：已购买并运行中（建议 2 核 2 GiB 及以上，系统盘 40 GiB）。
2. **操作系统**：Alibaba Cloud Linux 或 Ubuntu（本文以 Alibaba Cloud Linux 为例）。
3. **预装应用**：创建实例时可选「Docker」，若未选则下面会安装。
4. **安全组**：先记下，后面需放行 **TCP 8000** 端口。
5. **公网 IP**：记下实例公网 IP（如 `120.55.1.251`），用于浏览器访问。

---

## 二、连接服务器

在阿里云控制台点击实例的 **「远程连接」**，或本地使用 SSH：

```bash
ssh root@<公网IP>
# 示例：ssh root@120.55.1.251
# 若使用密钥或非 root 用户，请替换为实际用户名
```

---

## 三、安装 Docker 与 Docker Compose

若创建 ECS 时未预装 Docker，在 Alibaba Cloud Linux 上执行：

```bash
# 更新软件源
sudo yum update -y

# 安装 Docker（Alibaba Cloud Linux / CentOS 系）
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose 插件（Docker 官方方式）
sudo yum install -y docker-compose-plugin
# 若上面不可用，可改用：
# sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
# sudo chmod +x /usr/local/bin/docker-compose

# 验证
docker --version
docker compose version
```

若使用 **Ubuntu** 系统，可改用：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
```

---

## 四、安装 Git 并拉取代码

```bash
# 安装 Git（Alibaba Cloud Linux）
sudo yum install -y git

# 克隆仓库（替换为你的仓库地址）
cd /root
git clone https://github.com/lcx15228727985-jpg/dataset_platform.git
cd dataset_platform/dataset_viewer
```

---

## 五、准备数据目录

```bash
# 创建数据集根目录
sudo mkdir -p /data
sudo chown $USER:$USER /data

# 若有现成数据集，上传到 /data，结构示例：
# /data/
#   └── dataset0209/
#       ├── episode_0/
#       │   └── images_png/   # 或 images/
#       ├── episode_1/
#       │   └── images_png/
#       └── ...
# 暂无数据也可先建空目录，后续再上传
```

从本地上传数据集到服务器（在**本地 Windows** 执行，示例）：

```bash
scp -r D:\path\to\dataset\dataset0209 root@<公网IP>:/data/
```

---

## 六、构建并启动 Docker 容器

```bash
# 确保在 dataset_viewer 目录
cd /root/dataset_platform/dataset_viewer

# 设置数据根目录并构建、启动
export DATA_ROOT=/data
docker compose up -d --build
```

首次构建会拉取镜像并编译前端，约需数分钟。查看日志：

```bash
docker compose logs -f
```

按 `Ctrl+C` 退出日志，容器继续在后台运行。

---

## 七、初始化数据库

容器启动后，扫描数据集并建库（首次部署必做）：

```bash
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.init_pg_db 2>/dev/null || python -m backend.scan_dataset"
```

- 未配置 PostgreSQL 时，会使用 SQLite 执行 `scan_dataset`，将 run/episode/images 信息写入库。
- 若提示无数据，请确认 `/data` 下已有 `run/episode_N/images_png` 或 `images` 目录且内有图片。

---

## 八、放行 8000 端口（安全组）

1. 登录 **阿里云控制台** → **ECS** → 找到当前实例。
2. 点击 **安全组** → **配置规则** → **手动添加** 入方向规则：
   - **端口范围**：`8000/8000`
   - **授权对象**：`0.0.0.0/0`（或按需限制 IP）
   - **协议**：TCP
3. 保存。

---

## 九、验证部署

在服务器上检查：

```bash
curl http://localhost:8000/api/health
```

应返回类似：`{"ok":true,"database":"sqlite",...}`

在浏览器访问：

- **http://<公网IP>:8000**  
  例如：`http://120.55.1.251:8000`

能打开标注平台页面即表示部署成功。其他设备在同一网络或公网下也可通过该地址访问。

---

## 十、常用运维命令

```bash
# 进入项目目录
cd /root/dataset_platform/dataset_viewer

# 查看容器状态
docker ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 更新代码后重新构建并启动
git pull
export DATA_ROOT=/data
docker compose up -d --build

# 再次扫描数据集（新增 run/episode 后）
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.scan_dataset"

# 从磁盘 *_annot.json 同步到 DB（恢复备份后，使导出与统计正确）
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.sync_annotations_from_disk"
```

---

## 十一、标注备份与恢复（更新版本前后必读）

**备份标注（含 catalog.db + *_annot.json）：**
```bash
cd /data && find . \( -name "*_annot.json" -o -name "catalog.db" \) | tar -czvf /root/annot_backup_$(date +%Y%m%d_%H%M).tar.gz -T -
```

**下载到本机：**
```bash
scp "root@<公网IP>:/root/annot_backup_*.tar.gz" ~/Downloads/
```

**恢复标注（更新版本/重建后）：** 将备份解压到 `/data` 覆盖，然后：
```bash
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.scan_dataset"
# 若主标注页保存的标注需同步到 DB，再执行：
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.sync_annotations_from_disk"
```

---

## 十二、环境变量说明

| 变量 | 说明 | 默认 |
|------|------|------|
| DATA_ROOT | 数据集根目录（宿主机路径，挂载到容器 /data） | 本文使用 `/data` |
| EXPORT_DIR | 导出标注保存目录 | 容器内 `/data/export`，对应宿主机 `/data/export` |

导出标注（平台内「保存到服务器 (JSON）」）会写入 `/data/export/annotations.json`，训练脚本可直接从该路径读取。

---

## 十三、常见问题

### 13.1 端口 8000 无法访问

- 确认安全组已放行 **TCP 8000**（见第八步）。
- 在服务器上执行：`curl http://localhost:8000/api/health`，若正常则多为安全组或防火墙问题。

### 13.2 容器启动失败或构建报错

```bash
docker compose logs
```

查看具体错误。常见原因：磁盘不足、内存不足、网络拉取镜像失败。可重试：

```bash
docker compose up -d --build
```

### 12.3 平台显示「暂无 Run 数据」

- 确认 `/data` 下已有 run 目录，且内有 `episode_N/images_png` 或 `episode_N/images`。
- 执行第七步「初始化数据库」或再次执行：  
  `docker exec -it dataset-annotation bash -c "cd /app && python -m backend.scan_dataset"`

### 12.4 导出 404 或导出失败

- 确保已执行过第七步建库，且库中有数据。
- 检查：`curl http://localhost:8000/api/health` 与 `curl http://localhost:8000/api/deploy/status`。

### 12.5 修改端口（例如改为 9000）

编辑 `dataset_viewer/docker-compose.yml`，将 `ports` 改为：

```yaml
ports:
  - "9000:8000"
```

然后安全组放行 **9000**，访问 `http://<公网IP>:9000`。

---

## 十三、一键命令汇总（复制用）

以下为连贯步骤，在已连接 ECS 且为 root 时可直接按顺序执行（仓库地址请按需替换）：

```bash
# 1. 安装 Docker（Alibaba Cloud Linux）
sudo yum update -y
sudo yum install -y docker docker-compose-plugin
sudo systemctl start docker && sudo systemctl enable docker

# 2. 安装 Git 并拉取代码
sudo yum install -y git
cd /root
git clone https://github.com/lcx15228727985-jpg/dataset_platform.git
cd dataset_platform/dataset_viewer

# 3. 准备数据目录
sudo mkdir -p /data
sudo chown $USER:$USER /data

# 4. 构建并启动
export DATA_ROOT=/data
docker compose up -d --build

# 5. 初始化数据库
docker exec -it dataset-annotation bash -c "cd /app && python -m backend.init_pg_db 2>/dev/null || python -m backend.scan_dataset"

# 6. 验证
curl http://localhost:8000/api/health
```

最后在阿里云安全组放行 **TCP 8000**，浏览器访问 **http://<公网IP>:8000** 即可使用标注平台。
