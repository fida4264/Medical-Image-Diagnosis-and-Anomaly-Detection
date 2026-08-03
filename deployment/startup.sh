#!/usr/bin/env bash
# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Container Startup Script (Entrypoint)
# Module 22 — Docker, Deployment & Production Packaging
# ==============================================================================

set -eo pipefail

echo "======================================================================"
echo " Starting Medical AI Diagnosis & Anomaly Detection Platform"
echo " Version: v0.22.0.0"
echo " Environment: ${ENVIRONMENT:-production}"
echo "======================================================================"

# Ensure output, log, and database directories exist
mkdir -p /app/data /app/logs /app/outputs /app/saved_models

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

echo "[Startup] Initializing FastAPI REST API on http://${API_HOST}:${API_PORT}..."
uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" --log-level "${LOG_LEVEL:-info}" &
API_PID=$!

echo "[Startup] Initializing Gradio Web Application on http://0.0.0.0:${GRADIO_PORT}..."
python app.py &
GRADIO_PID=$!

# Trap signals for graceful shutdown
cleanup() {
    echo "[Shutdown] Caught termination signal. Gracefully shutting down services..."
    kill -TERM "$API_PID" 2>/dev/null || true
    kill -TERM "$GRADIO_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    wait "$GRADIO_PID" 2>/dev/null || true
    echo "[Shutdown] All services stopped."
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "======================================================================"
echo " Medical AI Platform is running:"
echo " • REST API:   http://localhost:${API_PORT}"
echo " • API Docs:   http://localhost:${API_PORT}/docs"
echo " • Gradio UI:  http://localhost:${GRADIO_PORT}"
echo "======================================================================"

# Wait for both processes to complete
wait "$API_PID" "$GRADIO_PID"
