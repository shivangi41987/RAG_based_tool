@echo off
echo ====================================
echo PDF RAG Assistant - Setup Script
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found
python --version
echo.

REM Create virtual environment
echo [2/4] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment and install dependencies
echo [3/4] Installing dependencies...
echo This may take 5-10 minutes on first run...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully
echo.

REM Success message
echo [4/4] Setup complete!
echo.
echo ====================================
echo SETUP SUCCESSFUL!
echo ====================================
echo.
echo To run the application:
echo   1. Open a new terminal
echo   2. Run: venv\Scripts\activate
echo   3. Run: streamlit run app.py
echo.
echo Or simply run: run_app.bat
echo.
echo Press any key to exit...
pause >nul
