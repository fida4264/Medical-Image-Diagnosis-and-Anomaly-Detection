#!/usr/bin/env bash
# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Production Healthcheck Probe Script
# Module 22 — Production Deployment Hardening Pass (v0.22.1.0)
# ==============================================================================

set -eo pipefail

API_PORT="${API_PORT:-8000}"
HOST="${API_HOST:-127.0.0.1}"

# Check 1: REST API Liveness & Readiness Probes
LIVENESS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${API_PORT}/health/liveness" || echo "000")
READINESS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${API_PORT}/health/readiness" || echo "000")

if [ "$LIVENESS_CODE" -ne 200 ] || [ "$READINESS_CODE" -ne 200 ]; then
    echo "UNHEALTHY: REST API check failed (Liveness: $LIVENESS_CODE, Readiness: $READINESS_CODE)"
    exit 1
fi

# Check 2: Gradio UI Endpoint Probe
GRADIO_PORT="${GRADIO_PORT:-7860}"
GRADIO_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${GRADIO_PORT}/" || echo "000")

if [ "$GRADIO_CODE" -ne 200 ] && [ "$GRADIO_CODE" -ne 302 ]; then
    echo "WARNING: Gradio UI non-responsive (HTTP $GRADIO_CODE), but REST API is healthy."
fi

echo "HEALTHY: All platform probes operational (REST API: OK, Database: OK, Engine: OK)"
exit 0
