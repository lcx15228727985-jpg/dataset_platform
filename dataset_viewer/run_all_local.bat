@echo off
chcp 65001 >nul
cd /d "%~dp0"
set SSLKEYLOGFILE=
set CONDA_NO_PLUGINS=1

echo ========== 1. 安装前端依赖 ==========
cd frontend
call npm install
if errorlevel 1 (
  echo npm install 失败，请检查 Node.js 环境
  pause
  exit /b 1
)
cd ..
echo.

echo ========== 2. 初始化数据库 ==========
if exist "%USERPROFILE%\Miniconda\envs\a2s\python.exe" (set "PY=%USERPROFILE%\Miniconda\envs\a2s\python.exe") else if exist "D:\Miniconda\envs\a2s\python.exe" (set "PY=D:\Miniconda\envs\a2s\python.exe") else (set "PY=python")

"%PY%" -m backend.init_pg_db 2>nul
if errorlevel 1 (
  echo PostgreSQL 不可用，改用 SQLite 扫描
  "%PY%" -m backend.scan_dataset
  if errorlevel 1 (
    echo 数据库初始化失败，请检查 DATA_ROOT 和 dataset 目录结构
    pause
    exit /b 1
  )
  echo 提示：标注工作台需 PostgreSQL，当前仅图库可用
) else (
  echo PostgreSQL 初始化成功，标注工作台可用
)
echo.

echo ========== 3. 启动后端 (端口 8000) ==========
start "Backend" cmd /k "cd /d %~dp0 && set SSLKEYLOGFILE= && set CONDA_NO_PLUGINS=1 && (if exist D:\Miniconda\envs\a2s\python.exe (D:\Miniconda\envs\a2s\python.exe) else python) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 4 /nobreak >nul

echo ========== 4. 启动前端 (端口 5173) ==========
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 3 /nobreak >nul
echo.
echo 部署完成。请在浏览器打开: http://localhost:5173
echo API 文档: http://localhost:8000/docs
echo 关闭本窗口不会停止服务，请到 Backend/Frontend 窗口关闭。
start http://localhost:5173
pause
