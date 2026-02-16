@echo off
echo ====================================
echo Starting PDF RAG Assistant...
echo ====================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to install dependencies.
    pause
    exit /b 1
)

REM Activate virtual environment and run app
call venv\Scripts\activate.bat
echo Starting Streamlit application...
echo.
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py
