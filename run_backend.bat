@echo off
cd /d "%~dp0"
echo [BACKEND] Starting Healthcare AI Assistant FastAPI Backend...
echo.
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please ensure the .venv folder exists in this directory.
    pause
    exit /b
)
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
