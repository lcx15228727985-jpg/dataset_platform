# 小组协作标注改造计划

在新一轮 git 提交与 Docker 阿里云部署前，需完成以下功能。

## 一、用户体系

### 1.1 用户账户
- **账号**：000 ~ 009（共 10 个）
- **权限**：全部为 root（等同管理员）
- **默认密码**：123456（暂不修改）
- **存储**：非关系型数据库（MongoDB）替代 SQLite，提升稳定性与并发能力

### 1.2 标注信息公开
- 10 个用户的标注数据全部公开，互相可见
- 支持 10 用户并发增删改查

---

## 二、登录与保存逻辑

### 2.1 登录入口
- 在**主页（图库）右上角**增加简易登录功能
- 登录后显示当前用户（如 003）

### 2.2 保存规则
- **仅登录用户的标注会被保存到服务端**
- 未登录时：可浏览、标注，但保存不写入数据库
- 登录后：Enter / 确认·保存 才会真正写入

---

## 三、技术架构

### 3.1 数据库
| 用途       | 原方案  | 新方案    |
|------------|---------|-----------|
| 用户       | 无      | MongoDB   |
| 标注数据   | SQLite/PostgreSQL | MongoDB（可选）或保留 PG |
| 图库索引   | PostgreSQL / SQLite | 保留 PG，SQLite 由 MongoDB 替代 |

### 3.2 认证
- JWT（JSON Web Token）存储用户身份
- 请求头 `Authorization: Bearer <token>` 传递
- 后端：`/api/auth/login` 登录，`/api/auth/me` 验证

### 3.3 Nginx 反向代理
- 提升网络稳定性与负载能力
- 对外端口 80/443，反向代理到后端 8000

---

## 四、实现清单

- [x] 后端：10 用户 000-009，密码 123456（auth.py）
- [x] 后端：JWT 登录接口 `/api/auth/login`
- [x] 后端：鉴权中间件，`/api/workstation/save` 需登录
- [x] 前端：主页、工作台右上角登录组件
- [x] 前端：API 携带 token，未登录时保存返回 401
- [x] Nginx 配置（反向代理）
- [x] Docker Compose 增加 Nginx 服务
- [x] MongoDB 用户存储（鲁棒性更好）
- [ ] 环境变量：`JWT_SECRET`（生产环境必改）

---

## 五、逐步实现指南（协作执行）

以下步骤可与你逐项完成，涉及下载、安装、注册、配置等。

---

### 步骤 1：JWT_SECRET 生产环境配置

**目的**：防止 token 被伪造，生产环境必须使用随机密钥。

| 序号 | 操作 | 说明 |
|------|------|------|
| 1.1 | 生成密钥 | 在服务器或本地执行：`openssl rand -hex 32`，得到 64 位十六进制串 |
| 1.2 | 保存密钥 | 将结果保存到 `.env` 或环境变量，切勿提交到 git |
| 1.3 | 配置 Docker | 在 `docker-compose up` 前执行：`export JWT_SECRET=<你的64位密钥>` |

**示例 `.env` 文件：**
```bash
# 在 dataset_viewer 目录下执行
cp .env.example .env
# 编辑 .env，将 JWT_SECRET 改为 openssl rand -hex 32 生成的 64 位字符串
```

---

### 步骤 2：MongoDB 用户存储（已实现）

**当前状态**：后端已集成 MongoDB 用户存储，优先从 MongoDB 读用户，无 MongoDB 时回退内存。

| 序号 | 操作 | 说明 |
|------|------|------|
| 2.1 | 本地开发 | `start.bat` 会自动尝试启动 MongoDB；或手动 `docker compose -f docker-compose.dev.yml up -d` |
| 2.2 | .env 配置 | `MONGODB_URI=mongodb://localhost:27017`（本地）或 `mongodb://mongo:27017`（Docker 部署） |
| 2.3 | 启动时 | 后端 startup 会自动 seed 10 用户 000-009 到 MongoDB |

---

### 步骤 3：阿里云部署流程（含 Nginx 80 端口）

| 序号 | 操作 | 说明 |
|------|------|------|
| 3.1 | 安全组 | 阿里云控制台 → ECS → 安全组 → 入方向规则，放行 **TCP 80**（Nginx） |
| 3.2 | 拉取代码 | `git pull` 或 `git clone` 到服务器 |
| 3.3 | 创建 `.env` | 在 `dataset_viewer` 目录新建 `.env`，填写 `JWT_SECRET`、`DATA_ROOT` |
| 3.4 | 启动服务 | `docker compose up -d --build` |
| 3.5 | 验证 | 浏览器访问 `http://<公网IP>`，应看到主页；登录 000/123456 后保存应成功 |

---

### 步骤 4：本地联调（可选）

| 序号 | 操作 | 说明 |
|------|------|------|
| 4.1 | 启动前后端 | `start.bat`（前端 5173 + 后端 8000） |
| 4.2 | 本地 Nginx 反向代理 | 安装 nginx 后运行 `start_nginx_local.bat`，统一入口 `http://localhost:8080` |
| 4.3 | 测试登录 | 打开 `http://localhost:8080`，右上角「登录」，输入 000 / 123456 |
| 4.4 | 测试保存 | 进入工作台，标注后按 Enter，应保存成功；退出登录后再保存应提示「请先登录」 |

---

## 六、部署流程（阿里云）快速参考

1. 拉取最新代码
2. 配置 `.env`：`JWT_SECRET`、`DATA_ROOT`
3. 安全组放行 TCP 80
4. `docker compose up -d`
5. 访问 `http://<公网IP>`，Nginx 监听 80，转发到后端 8000
