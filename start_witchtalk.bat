@echo off
setlocal

cd /d "%~dp0"
title WitchTalk Launcher
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "APP_FILE=%~dp0app.py"

if not exist "%PYTHON_EXE%" (
    echo [WitchTalk] Virtual environment not found: %PYTHON_EXE%
    echo [WitchTalk] Please create .venv and install dependencies first.
    echo.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo [WitchTalk] app.py was not found.
    echo [WitchTalk] Startup aborted.
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [WitchTalk] Warning: .env was not found in the project folder.
    echo [WitchTalk] Model features may not work until API settings are configured.
    echo.
)

echo [WitchTalk] Starting...
echo [WitchTalk] Working directory: %cd%
echo [WitchTalk] Python: %PYTHON_EXE%
echo [WitchTalk] Open http://127.0.0.1:5000 in your browser after startup.
echo.

"%PYTHON_EXE%" "%APP_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [WitchTalk] Process exited with code %EXIT_CODE%.
    echo [WitchTalk] If the window closed too fast before, please keep this window open and check the log above.
) else (
    echo [WitchTalk] Process exited normally.
)
pause
