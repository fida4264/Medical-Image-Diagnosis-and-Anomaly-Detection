# Installation Guide

## Medical Image Diagnosis & Anomaly Detection Platform (`v1.0.0`)

This document provides comprehensive, step-by-step instructions for installing, configuring, hardening, and running the Medical AI Platform on Windows, Linux, and macOS environments, with or without Docker.

---

## 📋 System Requirements

### Hardware Requirements
- **CPU:** Dual-core 2.0 GHz or higher (Quad-core recommended)
- **RAM:** Minimum 4 GB (8 GB recommended for multi-modality batch processing)
- **Disk Space:** 3 GB free space for application, models, dependencies, and database

### Software Prerequisites
- **Python:** Version 3.11 or higher (Python 3.11.x recommended)
- **Docker & Docker Compose:** Docker Engine 24.0+ & Docker Compose v2.20+ (Required for containerized deployment)
- **Operating Systems:** Windows 10/11 (WSL2 / PowerShell), Ubuntu 20.04/22.04 LTS, macOS Monterey (12.0+) or newer

---

## 🚀 Installation & Deployment Options

Choose one of the following setup options depending on your environment:

---

### Option A: Docker Containerized Installation (Recommended)

Docker provides a zero-configuration, reproducible, and security-hardened environment containing all necessary dependencies.

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/medical-image-diagnosis.git
cd medical-image-diagnosis
```

#### 2. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
*(Optional: Edit `.env` to configure ports, API keys, resource limits `CPU_LIMIT=2.0`, `MEMORY_LIMIT=4G`, or logging levels).*

#### 3. Build & Launch Containers using Profiles
To run the production profile:
```bash
docker compose --profile production up --build -d
```
Or for default stack:
```bash
docker compose up --build -d
```

#### 4. Verify Container Health
```bash
docker compose ps
docker compose logs -f
```

Access the platform at:
- **Gradio Web Dashboard:** [http://localhost:7860](http://localhost:7860)
- **FastAPI REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Bare-Metal Installation (Windows PowerShell / CMD)

#### 1. Verify Python Installation
Open PowerShell / Command Prompt and verify Python 3.11+:
```powershell
python --version
```

#### 2. Clone Repository & Navigate
```powershell
git clone https://github.com/your-username/medical-image-diagnosis.git
cd "medical-image-diagnosis"
```

#### 3. Run Automated Launcher
Run the launcher script:
```powershell
.\run.bat
```
*(Or in Command Prompt: `run.bat`)*

---

### Option C: Bare-Metal Installation (Linux / macOS)

#### 1. Clone Repository
```bash
git clone https://github.com/your-username/medical-image-diagnosis.git
cd medical-image-diagnosis
```

#### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run Launcher Script
Make `run.sh` executable and run it:
```bash
chmod +x run.sh deployment/*.sh
./run.sh
```

---

## 🛠️ Environment Variables Configuration

Refer to `.env.example` for all configurable variables:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Environment mode (`development`, `testing`, `production`) |
| `AUTH_REQUIRED` | `false` | Requires `X-API-Key` header if true |
| `API_KEYS` | (empty) | Comma-separated list of valid API keys |
| `API_PORT` | `8000` | Port for FastAPI REST API |
| `GRADIO_PORT` | `7860` | Port for Gradio Web UI |
| `LOG_LEVEL` | `info` | Console logging detail (`debug`, `info`, `warning`, `error`) |
| `DATABASE_PATH` | `data/medical_ai.db` | SQLite database file location |
| `CPU_LIMIT` | `2.0` | Maximum CPU cores allocated to container |
| `MEMORY_LIMIT` | `4G` | Maximum RAM memory limit allocated to container |

---

## 🧹 Docker Resource Management & Cleanup

To clean unused Docker resources, images, or volumes:
```bash
# Stop running containers
docker compose down

# Remove unused volumes and images
docker system prune -a --volumes
```

---

## ❓ Troubleshooting & Common Issues

### Issue 1: Port Conflict (8000 or 7860 in use)
**Solution:** Change ports in `.env` or pass environment variables:
```bash
API_PORT=8080 GRADIO_PORT=7870 ./run.sh
```

### Issue 2: `PowerShell script / bat execution policy` (Windows)
**Solution:** Run using `.\run.bat` in PowerShell or Command Prompt (`run.bat`).

### Issue 3: Read-Only Filesystem Permissions
**Solution:** Mounted directories (`/app/data`, `/app/logs`, `/app/outputs`) are configured with non-root ownership (`medai:1001`). Ensure host volume paths have appropriate write permissions.
