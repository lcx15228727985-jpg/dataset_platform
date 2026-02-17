@echo off
cd /d "%~dp0"
set SSLKEYLOGFILE=
set CONDA_NO_PLUGINS=1
if exist "%USERPROFILE%\Miniconda\envs\a2s\python.exe" (set "PY=%USERPROFILE%\Miniconda\envs\a2s\python.exe") else if exist "D:\Miniconda\envs\a2s\python.exe" (set "PY=D:\Miniconda\envs\a2s\python.exe") else (set "PY=python")
echo Running scan_dataset...
"%PY%" -m backend.scan_dataset
echo.
pause
