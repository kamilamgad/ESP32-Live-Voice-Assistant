@echo off
setlocal
cd /d "%~dp0pc_server"
"%~dp0.venv\Scripts\python.exe" -m uvicorn server:app --host 0.0.0.0 --port 8000
