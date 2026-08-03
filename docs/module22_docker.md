# Module 22 — Docker Architecture & Production Deployment Documentation

## Medical Image Diagnosis & Anomaly Detection Platform (`v0.22.1.0`)

---

## 📌 Architecture Overview

Module 22 packages the entire Medical AI Platform into a portable, production-grade containerized environment and provides automated local cross-platform deployment tooling.

```mermaid
graph TD
    Client[Client / Clinician Browser] -->|Port 7860| Gradio[Gradio UI Dashboard]
    Client -->|Port 8000| API[FastAPI REST API]
    
    subgraph Container ["Docker Container (Hardened medai user: 1001)"]
        Gradio --> Engine[MedicalAIEngine Core]
        API --> Engine
        Engine --> Modality[Modality Classifier]
        Engine --> Disease[7 Disease Predictors]
        Engine --> XAI[GradCAM XAI Engine]
        Engine --> Anomaly[OOD Anomaly Detector]
        Engine --> DB[(SQLite Database /app/data)]
        Engine --> Reports[Report Generator Engine]
    end

    subgraph Host Storage (Persistent Volumes)
        DB Volume[(medai_data)] --> DB
        Logs Volume[(medai_logs)] --> Logs[/app/logs]
        Outputs Volume[(medai_outputs)] --> Outputs[/app/outputs]
    end
```

---

## 🐳 Docker Multi-Stage & OCI Metadata Strategy

The production `docker/Dockerfile` utilizes a two-stage build to minimize image size and eliminate build tool security vulnerabilities:

1. **Stage 1 (`builder`):**
   - Base image: `--platform=${TARGETPLATFORM} python:3.11-slim`
   - Installs build utilities (`gcc`, `build-essential`)
   - Builds isolated Python virtualenv (`/opt/venv`) with CPU-only PyTorch and all dependencies.

2. **Stage 2 (`runtime`):**
   - Base image: `python:3.11-slim`
   - OCI-Compliant Metadata Labels (`org.opencontainers.image.*`)
   - Copies `/opt/venv` from Stage 1 (omitting compilers)
   - Creates restricted non-root user `medai` (UID: 1001, GID: 1001)
   - Copies application code (`api/`, `medical_ai_engine/`, `app.py`, `deployment/`, `configs/`)
   - Exposes ports `8000` (FastAPI) and `7860` (Gradio)
   - Configures healthcheck probe via `deployment/healthcheck.sh`

---

## 🔒 Container Security Hardening

| Hardening Feature | Configuration | Purpose |
|---|---|---|
| Non-Root User | `USER medai:1001` | Prevents root privilege escalation inside container |
| Read-Only Root Filesystem | `read_only: true` | Prevents unauthorized file system modification |
| Temporary Filesystem | `tmpfs: /tmp` (100M) | Provides secure in-memory scratch space |
| Capability Dropping | `cap_drop: [ALL]` | Drops all Linux kernel capabilities |
| Privilege Escalation Guard | `security_opt: [no-new-privileges:true]` | Blocks SUID binary execution |
| Log Rotation | `json-file` (10m max, 5 files) | Prevents container disk space exhaustion |

---

## ⚙️ Resource Management & Container Compose Profiles

### Resource Caps (Configurable via `.env`)
- **CPU Limits:** `2.0` cores (Reservation: `0.5` cores)
- **Memory Limits:** `4 GB` RAM (Reservation: `1 GB` RAM)

### Compose Profiles
- **`production`**: Production multi-stage build with full security restrictions.
- **`development`**: Hot-reloading dev container mounting local source folders.
- **`testing`**: Automated containerized test execution context.

To launch a specific profile:
```bash
docker compose --profile production up -d
docker compose --profile development up -d
docker compose --profile testing up
```

---

## 🏷️ Docker Image Versioning & Tagging Convention

Recommended image tags for production release pipelines:
- `medical-ai-platform:v0.22.1.0` (Exact release version)
- `medical-ai-platform:latest` (Latest stable release pointer)
- `medical-ai-platform:dev` (Development branch build)

Example build command:
```bash
docker build -t medical-ai-platform:v0.22.1.0 -t medical-ai-platform:latest -f docker/Dockerfile .
```

---

## ☁️ Kubernetes Migration Readiness

A production Kubernetes manifest template is provided at [`configs/k8s_deployment_template.yaml`](file:///d:/Medical%20Image%20Diagnosis%20and%20Anomaly%20Detection/configs/k8s_deployment_template.yaml) featuring:
- Dedicated Namespace (`medical-ai`)
- SecurityContext with `runAsUser: 1001` and `readOnlyRootFilesystem: true`
- ConfigMap for environment settings
- PersistentVolumeClaim (`medai-pvc-data`) for database durability
- Liveness & Readiness Probes mapped to port 8000
