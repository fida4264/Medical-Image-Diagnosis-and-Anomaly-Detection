@echo off
rem ==============================================================================
rem Medical Image Diagnosis & Anomaly Detection Platform
rem Local Windows Batch Launcher
rem Module 22 — Docker, Deployment & Production Packaging
rem ==============================================================================

echo ======================================================================
echo  Medical Image Diagnosis ^& Anomaly Detection Platform
echo  Windows Local Deployment Launcher
echo ======================================================================

rem 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python is not installed or not added to PATH.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

echo [Check] Python detected successfully.

rem 2. Create required directories
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs
if not exist "data" mkdir data
if not exist "saved_models" mkdir saved_models
echo [Check] Runtime folders created/verified.

rem 3. Verify packages
python -c "import fastapi, gradio, torch, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo [Notice] Installing missing Python requirements...
    python -m pip install -r requirements.txt
)

rem 4. Set default ports
if "%API_PORT%"=="" set API_PORT=8000
if "%GRADIO_PORT%"=="" set GRADIO_PORT=7860

echo ======================================================================
echo  Launching Services...
echo ======================================================================

start "Medical AI REST API" cmd /k "python -m uvicorn api.main:app --host 127.0.0.1 --port %API_PORT%"
start "Medical AI Gradio UI" cmd /k "python app.py"

echo ======================================================================
echo  Platform Launched Successfully!
echo  • REST API:      http://127.0.0.1:%API_PORT%
echo  • API Docs:      http://127.0.0.1:%API_PORT%/docs
echo  • Gradio UI:     http://127.0.0.1:%GRADIO_PORT%
echo.
echo  Close the popup command windows to stop the servers.
echo ======================================================================
