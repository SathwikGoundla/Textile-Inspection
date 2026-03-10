@echo off
echo ============================================
echo   TextileVision AI - Quality Inspection
echo ============================================
echo.

cd /d "%~dp0backend"

echo [1/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Starting backend server...
echo.
echo   Dashboard: http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
