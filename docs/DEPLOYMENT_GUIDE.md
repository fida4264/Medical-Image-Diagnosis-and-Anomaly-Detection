# Deployment Guide
## Medical Image Diagnosis and Anomaly Detection Platform
### Version 0.22.1.0

---

## Table of Contents

1. [Quick Reference](#1-quick-reference)
2. [Prerequisites](#2-prerequisites)
3. [Local Development Setup](#3-local-development-setup)
4. [Docker Production Deployment](#4-docker-production-deployment)
5. [Kubernetes Deployment](#5-kubernetes-deployment)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Reverse Proxy Setup (Nginx)](#7-reverse-proxy-setup-nginx)
8. [Health Checks & Monitoring](#8-health-checks--monitoring)
9. [Backup & Recovery](#9-backup--recovery)
10. [Upgrading](#10-upgrading)

---

## 1. Quick Reference

| Deployment Mode | Command | Use Case |
|---|---|---|
| Windows Local | `run.bat` | Development / Demo |
| Linux/macOS Local | `./run.sh` | Development / Demo |
| Docker Production | `docker compose --profile production up -d` | Production |
| Docker Development | `docker compose --profile development up` | Development |
| Kubernetes | `kubectl apply -f deployment/kubernetes/` | Cloud / Enterprise |

---

## 2. Prerequisites

| Tool | Version | Required For |
|---|---|---|
| Python | 3.9+ | Local development |
| pip | 22+ | Local development |
| Docker | 20.10+ | Docker deployment |
| Docker Compose | 2.x | Docker deployment |
| kubectl | 1.24+ | Kubernetes deployment |

---

## 3. Local Development Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd "Medical Image Diagnosis and Anomaly Detection"

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set API keys and preferences

# 5. Launch all services
python app.py                                  # Gradio → port 7860
python -m uvicorn api.main:app --reload        # API → port 8000
```

---

## 4. Docker Production Deployment

### 4.1 Initial Setup

```bash
# Configure environment variables
cp .env.example .env
nano .env  # or: notepad .env (Windows)

# Required: set at least one API key
MEDAI_API_KEY_1=your-secure-api-key-here
```

### 4.2 Build and Launch

```bash
# Production deployment
docker compose --profile production up --build -d

# Check container status
docker compose ps

# View logs
docker compose logs -f medical-ai-platform
```

### 4.3 Verify Deployment

```bash
# Check liveness
curl http://localhost:8000/health/liveness

# Check readiness
curl http://localhost:8000/health/readiness

# Test prediction (replace YOUR_KEY)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@test_image.jpg" \
  -F "modality=chest_xray"
```

### 4.4 Stop / Restart

```bash
# Stop services
docker compose --profile production down

# Restart with rebuild
docker compose --profile production up --build -d

# View resource usage
docker stats medical-ai-platform
```

---

## 5. Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f deployment/kubernetes/

# Check pod status
kubectl get pods -l app=medical-ai-platform

# View logs
kubectl logs -l app=medical-ai-platform -f

# Check service
kubectl get service medical-ai-service

# Scale deployment
kubectl scale deployment medical-ai-platform --replicas=3
```

**Services exposed:**
- Port 8000 → FastAPI REST API
- Port 7860 → Gradio Dashboard

---

## 6. Environment Variables Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `API_PORT` | int | `8000` | FastAPI port |
| `GRADIO_PORT` | int | `7860` | Gradio port |
| `AUTH_REQUIRED` | bool | `true` | Enable API key auth |
| `MEDAI_API_KEY_1` | str | — | Primary API key |
| `MEDAI_API_KEY_2` | str | — | Secondary API key |
| `RATE_LIMIT_PREDICT` | int | `20` | Predict limit/minute |
| `RATE_LIMIT_HISTORY` | int | `60` | History limit/minute |
| `MAX_FILE_SIZE_MB` | int | `20` | Upload limit in MB |
| `LOG_LEVEL` | str | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `DB_PATH` | str | `data/predictions.db` | SQLite database path |
| `REPORTS_DIR` | str | `outputs/` | Report output directory |
| `ALLOWED_ORIGINS` | str | `*` | CORS allowed origins |

---

## 7. Reverse Proxy Setup (Nginx)

Place the following in your Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # FastAPI REST API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 25M;
    }

    # Gradio Dashboard
    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 8. Health Checks & Monitoring

### Health Endpoints

| Endpoint | Use |
|---|---|
| `GET /health/liveness` | Kubernetes liveness probe |
| `GET /health/readiness` | Kubernetes readiness probe |
| `GET /system/info` | Version and status |

### Log Locations

| Location | Description |
|---|---|
| `logs/api.log` | FastAPI request/response logs |
| `logs/engine.log` | AI engine prediction logs |
| `logs/anomaly.log` | OOD detection events |
| Docker: `docker compose logs` | Container stdout/stderr |

---

## 9. Backup & Recovery

### Database Backup

```bash
# Manual backup
cp data/predictions.db data/predictions_backup_$(date +%Y%m%d).db

# Automated backup (using provided script)
./deployment/scripts/backup.sh
```

### Volume Backup (Docker)

```bash
# Backup all volumes
docker run --rm -v medai_data:/source -v $(pwd)/backup:/dest \
  alpine tar czf /dest/medai_data_backup.tar.gz -C /source .
```

---

## 10. Upgrading

```bash
# Pull latest code
git pull origin main

# Rebuild and restart (Docker)
docker compose --profile production up --build -d

# Database migrations (if required)
python -m alembic upgrade head
```

Always check [CHANGELOG.md](../CHANGELOG.md) for breaking changes before upgrading.
