@echo off
cd /d "%~dp0"

echo Current dir: %CD%
if not exist "frontend\package.json" (
  echo Error: frontend\package.json not found. Run from dataset_viewer folder.
  pause
  exit /b 1
)

echo.
echo [1/6] Installing frontend deps...
cd frontend
call npm install
if errorlevel 1 (
  echo npm install failed
  pause
  exit /b 1
)
cd ..

echo.
echo [2/6] JWT local env...
if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example
  )
) else (
  echo .env exists
)

echo.
echo [3/6] MongoDB...
docker compose -f docker-compose.dev.yml up -d 2>nul
if errorlevel 1 (
  echo Docker/MongoDB skipped, using in-memory users
) else (
  echo MongoDB started on localhost:27017
  timeout /t 2 /nobreak >nul
)

echo.
echo [4/6] Init database...
if exist "%USERPROFILE%\Miniconda\envs\a2s\python.exe" (set "PY=%USERPROFILE%\Miniconda\envs\a2s\python.exe") else if exist "D:\Miniconda\envs\a2s\python.exe" (set "PY=D:\Miniconda\envs\a2s\python.exe") else (set "PY=python")
"%PY%" -m backend.init_pg_db 2>nul
if errorlevel 1 (
  "%PY%" -m backend.scan_dataset 2>nul
  if errorlevel 1 echo DB init failed, gallery may be limited
) else (
  echo DB init OK
)

echo.
echo [5/6] Starting backend on :8000...
start "Backend" cmd /k "cd /d %~dp0 & set PYTHONPATH=%~dp0 & set SSLKEYLOGFILE= & "%PY%" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 4 /nobreak >nul

echo.
echo [6/6] Starting frontend on :5173...
start "Frontend" cmd /k "cd /d %~dp0frontend & npm run dev"

timeout /t 3 /nobreak >nul
echo.
echo Done. Open http://localhost:5173
echo API docs: http://localhost:8000/docs
echo Close Backend/Frontend windows to stop services.
start http://localhost:5173
pause
