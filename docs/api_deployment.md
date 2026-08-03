# Medical AI Platform — API Production Deployment Guide

## Overview

This guide details operational procedures for deploying the **Medical Image Diagnosis & Anomaly Detection REST API** (Module 21) in enterprise production environments.

---

## Deployment Architecture

```
[ Internet / Internal VNet ]
             │
             ▼
    ┌─────────────────┐
    │   Nginx Reverse │  (TLS Termination, Rate Limit, Buffer)
    │      Proxy      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Gunicorn Master │  (Uvicorn Worker Processes)
    │  (App Server)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  FastAPI REST   │  (Module 21 Engine)
    └─────────────────┘
```

---

## 1. Production Configuration Checklist

Before deploying, verify environment settings in `/etc/medical_ai.env` or systemd environment file:

```ini
ENVIRONMENT=production
DEBUG=false

# Require strict authentication
AUTH_REQUIRED=true
API_KEYS=prod_secure_key_884920192847,prod_service_key_119284019283

# Disable public Swagger documentation in production
ENABLE_DOCS=false

# Configure allowed CORS origins (NO WILDCARD '*')
ALLOWED_CORS_ORIGINS=https://medical-ai.hospital.org,https://dashboard.hospital.org

# File size & rate limits
MAX_UPLOAD_SIZE_BYTES=20971520
RATE_LIMIT_PREDICT_PER_MIN=20
```

---

## 2. Gunicorn + Uvicorn Production Launch Command

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --timeout 120 \
  --access-logfile /var/log/medical_ai/access.log \
  --error-logfile /var/log/medical_ai/error.log
```

---

## 3. Systemd Service Setup (`/etc/systemd/system/medical-ai-api.service`)

```ini
[Unit]
Description=Medical AI Platform REST API Service
After=network.target

[Service]
User=medai
Group=medai
WorkingDirectory=/opt/medical_ai_platform
EnvironmentFile=/etc/medical_ai.env
ExecStart=/opt/medical_ai_platform/venv/bin/gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 4. Kubernetes Health Probes

Configure pod readiness and liveness probes using Module 21 system endpoints:

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 5
```
