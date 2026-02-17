@echo off
cd /d "%~dp0"

REM 需先运行 start.bat 启动前端(5173)和后端(8000)，再运行本脚本
echo Ensure frontend (5173) and backend (8000) are running (start.bat)...
echo.

REM 创建 nginx 所需临时目录（-c 使用配置文件所在目录为 prefix）
if not exist "temp" mkdir temp
if not exist "temp\client_body_temp" mkdir temp\client_body_temp
if not exist "temp\proxy_temp" mkdir temp\proxy_temp
if not exist "temp\fastcgi_temp" mkdir temp\fastcgi_temp

REM 检测 nginx 是否已运行
tasklist /fi "imagename eq nginx.exe" 2>nul | find /i "nginx.exe" >nul
if %errorlevel%==0 (
  echo Nginx is already running. Stopping first...
  nginx -s stop 2>nul
  timeout /t 2 /nobreak >nul
)

REM 配置文件路径（不用 -p，避免 Windows 路径解析问题）
set "CONF=%~dp0nginx.local.conf"

REM 检查 nginx 是否安装
where nginx >nul 2>&1
if %errorlevel% neq 0 (
  echo Nginx not found. Please install nginx and add to PATH.
  echo Or download from https://nginx.org/en/download.html
  echo Extract and add nginx.exe directory to PATH.
  pause
  exit /b 1
)

REM 先启动 start.bat 里的前后端，再启动 nginx
echo Starting nginx with local reverse proxy (port 8080)...
echo Frontend: 127.0.0.1:5173
echo Backend:  127.0.0.1:8000
echo Access:   http://localhost:8080
echo.

nginx -c "%CONF%"
if %errorlevel% neq 0 (
  echo Nginx failed to start.
  pause
  exit /b 1
)

echo Nginx started. Open http://localhost:8080
echo Press any key to stop nginx...
pause >nul
nginx -s stop
echo Nginx stopped.
