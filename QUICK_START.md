# ⚡ Quick Start Guide

## Medical Image Diagnosis & Anomaly Detection Platform (`v1.0.0`)

Get the entire Medical AI Platform running in under **5 minutes**.

---

## ⚡ Option 1: Docker (Fastest — 1 Command)

### 1. Clone & Enter Directory
```bash
git clone https://github.com/your-username/medical-image-diagnosis.git
cd medical-image-diagnosis
```

### 2. Start Production Stack
```bash
docker compose --profile production up --build -d
```

### 3. Open Platform
- 🎨 **Gradio Web Interface:** [http://localhost:7860](http://localhost:7860)
- 🔌 **FastAPI REST API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Stop Stack
```bash
docker compose down
```

---

## ⚡ Option 2: Local Bare-Metal (No Docker)

### Windows (PowerShell / Command Prompt)
Run in PowerShell:
```powershell
.\run.bat
```
*(Or in Command Prompt: `run.bat`)*

### Linux / macOS
Run in Terminal:
```bash
chmod +x run.sh deployment/*.sh
./run.sh
```

---

## 🔍 Quick API Verification

Check service health via `curl`:
```bash
curl http://localhost:8000/health/liveness
```

Expected response:
```json
{"status": "UP"}
```

Check deep readiness probe:
```bash
curl http://localhost:8000/health/readiness
```

---

## 🛑 Stopping & Cleaning Up

- **Stop Docker Stack:** `docker compose down`
- **Clean Docker Build Caches:** `docker system prune -a`
- **Local Bare-Metal:** Press `Ctrl+C` in the terminal windows.
