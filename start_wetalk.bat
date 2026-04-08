@echo off
setlocal

cd /d "%~dp0"
title Wetalk Launcher

if not exist ".venv\Scripts\python.exe" (
    echo [Wetalk] 未找到虚拟环境：.venv\Scripts\python.exe
    echo [Wetalk] 请先创建虚拟环境并安装依赖。
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo [Wetalk] 未找到 app.py，无法启动项目。
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [Wetalk] 警告：当前目录下未找到 .env 文件。
    echo [Wetalk] 如果没有配置 API Key，模型相关功能可能无法正常使用。
    echo.
)

echo [Wetalk] 正在启动...
echo [Wetalk] 项目目录：%cd%
echo.

call ".venv\Scripts\activate.bat"
python app.py

echo.
echo [Wetalk] 程序已退出。
pause
