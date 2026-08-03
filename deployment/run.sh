#!/usr/bin/env bash
# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Local Runner Script (Linux / macOS)
# Module 22 — Docker, Deployment & Production Packaging
# ==============================================================================

set -eo pipefail

echo "======================================================================"
echo " Medical Image Diagnosis & Anomaly Detection Platform"
echo " Bare-Metal / Local Host Launcher"
echo "======================================================================"

# 1. Validate Python version
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    PYTHON_CMD="python"
fi

if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[Error] Python 3 is not installed or not in PATH."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[Check] Python version: $PY_VER"

# 2. Check & create required directories
echo "[Check] Creating required runtime folders..."
mkdir -p logs outputs data saved_models

# 3. Check for dependencies
echo "[Check] Verifying Python dependencies..."
if ! $PYTHON_CMD -c "import fastapi, gradio, torch, PIL" 2>/dev/null; then
    echo "[Notice] Missing required dependencies. Installing from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# 4. Load environment variables if .env exists
if [ -f .env ]; then
    echo "[Env] Loading environment configuration from .env..."
    export $(grep -v '^#' .env | xargs)
fi

API_PORT="${API_PORT:-8000}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

echo "======================================================================"
echo " Launching Services..."
echo "======================================================================"

echo "[Launch] Starting FastAPI REST API..."
$PYTHON_CMD -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

echo "[Launch] Starting Gradio UI Dashboard..."
$PYTHON_CMD app.py &
GRADIO_PID=$!

cleanup() {
    echo ""
    echo "[Stopping] Terminating background processes..."
    kill "$API_PID" 2>/dev/null || true
    kill "$GRADIO_PID" 2>/dev/null || true
    echo "[Stopped] Platform stopped successfully."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "======================================================================"
echo " Platform Available At:"
echo " • REST API:      http://127.0.0.1:${API_PORT}"
echo " • API Docs:      http://127.0.0.1:${API_PORT}/docs"
echo " • Gradio UI:     http://127.0.0.1:${GRADIO_PORT}"
echo " Press Ctrl+C to stop all services."
echo "======================================================================"

wait "$API_PID" "$GRADIO_PID"
