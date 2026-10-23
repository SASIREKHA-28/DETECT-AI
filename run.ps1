# DetectAI PowerShell Launcher
Set-Location $PSScriptRoot
Write-Host "Activating Virtual Environment..." -ForegroundColor Cyan
& "..\deepfake-detector\venv\Scripts\Activate.ps1"
Write-Host "Launching DetectAI on http://127.0.0.1:8000..." -ForegroundColor Green
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
