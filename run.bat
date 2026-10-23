@echo off
cd /d "%~dp0"
echo Starting DetectAI Web App...
call "..\deepfake-detector\venv\Scripts\activate.bat"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
