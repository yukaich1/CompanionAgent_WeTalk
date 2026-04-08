@echo off
setlocal

cd /d "%~dp0"
title Wetalk Launcher

if not exist ".venv\Scripts\python.exe" (
    echo [Wetalk] Virtual environment not found: .venv\Scripts\python.exe
    echo [Wetalk] Please create .venv and install dependencies first.
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [Wetalk] app.py was not found.
    echo [Wetalk] Startup aborted.
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [Wetalk] Warning: .env was not found in the project folder.
    echo [Wetalk] Model features may not work until API settings are configured.
    echo.
)

echo [Wetalk] Starting...
echo [Wetalk] Working directory: %cd%
echo.

call ".venv\Scripts\activate.bat"
python app.py

echo.
echo [Wetalk] Process exited.
pause
