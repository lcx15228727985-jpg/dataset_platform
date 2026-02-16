@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 数据根目录：优先用环境变量 DATA_ROOT，否则用 IROS/dataset
set "DATA_ROOT=%DATA_ROOT%"
if "%DATA_ROOT%"=="" set "DATA_ROOT=%~dp0..\dataset"

echo 数据根目录: %DATA_ROOT%
echo.
echo 将删除各 episode 下的 images（仅当存在 images_png 时），按任意键执行...
pause >nul

if exist "%USERPROFILE%\Miniconda\envs\a2s\python.exe" (set "PY=%USERPROFILE%\Miniconda\envs\a2s\python.exe") else if exist "D:\Miniconda\envs\a2s\python.exe" (set "PY=D:\Miniconda\envs\a2s\python.exe") else (set "PY=python")

"%PY%" remove_images_keep_png.py "%DATA_ROOT%"
echo.
pause
