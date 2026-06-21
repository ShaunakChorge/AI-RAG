@echo off
cd /d "%~dp0"
echo [FRONTEND] Starting Healthcare AI Assistant Streamlit UI...
echo.
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please ensure the .venv folder exists in this directory.
    pause
    exit /b
)
call .venv\Scripts\activate.bat
python -m streamlit run streamlit_app.py --server.port 8501
pause
