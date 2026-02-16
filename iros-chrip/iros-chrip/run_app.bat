@echo off
REM Unset SSL keylog to avoid PermissionError when starting Streamlit
set SSLKEYLOGFILE=
set SSLKEYLOGFILE=

cd /d "%~dp0"
echo Starting iros-chrip Streamlit app...
python -m streamlit run app.py --server.headless true
pause
