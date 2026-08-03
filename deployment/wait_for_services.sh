#!/usr/bin/env bash
# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Service Readiness Wait Script
# Module 22 — Docker, Deployment & Production Packaging
# ==============================================================================

set -eo pipefail

API_PORT="${API_PORT:-8000}"
HOST="${API_HOST:-127.0.0.1}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "[Wait] Waiting for Medical AI REST API to become ready on http://${HOST}:${API_PORT}/health/readiness..."

count=0
until [ "$count" -ge "$MAX_RETRIES" ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${API_PORT}/health/readiness" || echo "000")
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "[Ready] Medical AI REST API is ready!"
        exit 0
    fi
    count=$((count + 1))
    echo "[Wait] Attempt $count/$MAX_RETRIES failed (HTTP $HTTP_CODE). Retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

echo "[Error] Service failed to become ready after $((MAX_RETRIES * RETRY_INTERVAL)) seconds."
exit 1
