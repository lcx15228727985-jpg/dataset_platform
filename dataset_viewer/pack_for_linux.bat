@echo off
chcp 65001 >nul
cd /d "%~dp0"

set OUT=dataset_viewer_linux.zip
set TMP=pack_tmp
if exist "%OUT%" del "%OUT%"
if exist "%TMP%" rmdir /s /q "%TMP%"
mkdir "%TMP%"

echo 正在打包 Linux 部署包（排除 node_modules, __pycache__, dist）...
robocopy backend "%TMP%\backend" /E /XD __pycache__ .git /NFL /NDL /NJH /NJS /nc /ns /np
robocopy frontend "%TMP%\frontend" /E /XD node_modules __pycache__ dist .git /NFL /NDL /NJH /NJS /nc /ns /np
copy Dockerfile "%TMP%\" >nul
copy docker-compose.yml "%TMP%\" >nul
copy .dockerignore "%TMP%\" 2>nul
copy LINUX_DEPLOY.md "%TMP%\" 2>nul
copy ACCESS_GUIDE.md "%TMP%\" 2>nul

powershell -NoProfile -Command "Compress-Archive -Path '%TMP%\*' -DestinationPath '%OUT%' -Force -CompressionLevel Optimal"
rmdir /s /q "%TMP%"
if not exist "%OUT%" (
  echo 打包失败，请手动将 backend、frontend、Dockerfile、docker-compose.yml 复制后打包（排除 node_modules、__pycache__、dist）
  pause
  exit /b 1
)

echo.
echo 部署包已生成: %OUT%
echo 传到 Linux 后: unzip %OUT% -d dataset_viewer ^&^& cd dataset_viewer ^&^& docker-compose up -d --build
pause
