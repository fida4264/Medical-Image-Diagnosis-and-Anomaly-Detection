# Developer Manual
## Medical Image Diagnosis and Anomaly Detection Platform
### Version 0.22.1.0

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Core Components](#4-core-components)
5. [Adding a New Modality Predictor](#5-adding-a-new-modality-predictor)
6. [REST API Internals](#6-rest-api-internals)
7. [Database Schema](#7-database-schema)
8. [Configuration System](#8-configuration-system)
9. [Testing](#9-testing)
10. [Docker & Deployment](#10-docker--deployment)
11. [Code Style & Contribution Guidelines](#11-code-style--contribution-guidelines)
12. [Module Architecture Reference](#12-module-architecture-reference)

---

## 1. Architecture Overview

The platform uses a **layered, modular architecture** with strict separation of concerns:

```
┌──────────────────────────────────────────────────────────┐
│          Presentation Layer                               │
│   Gradio Web Dashboard (app.py) | FastAPI REST API        │
├──────────────────────────────────────────────────────────┤
│          Orchestration Layer                              │
│          MedicalAIEngine (core/engine.py)                 │
├───────────────┬──────────────────────┬───────────────────┤
│  Prediction   │   Explainability     │  Anomaly Detection │
│  Layer        │   Layer (XAI)        │  Layer (OOD)       │
│ core/models/  │  core/xai/           │  core/anomaly/     │
├───────────────┴──────────────────────┴───────────────────┤
│          Persistence Layer                                │
│     SQLite (database.py) + ReportGenerator               │
├──────────────────────────────────────────────────────────┤
│          Infrastructure Layer                             │
│   Configuration | Logging | Metrics | Docker              │
└──────────────────────────────────────────────────────────┘
```

**Key design principles:**
- `MedicalAIEngine` is the **single orchestration point** — no caller should invoke individual subsystems directly
- All API endpoints are **thin wrappers** over engine calls — no business logic in route handlers
- Modality predictors are **plugins** — new modalities require no changes to the engine

---

## 2. Repository Structure

```
Medical Image Diagnosis and Anomaly Detection/
├── api/                          # FastAPI REST API (Module 21)
│   ├── main.py                   # ASGI app, middleware registration
│   ├── routes/                   # Route handlers (thin wrappers)
│   │   ├── prediction.py
│   │   ├── history.py
│   │   ├── reports.py
│   │   ├── benchmark.py
│   │   └── system.py
│   ├── schemas.py                # Pydantic request/response models
│   ├── dependencies.py           # Dependency injection (auth, engine)
│   ├── middleware.py             # Security, tracing, rate limiting
│   ├── exceptions.py             # Unified error handling
│   └── tests/                   # API integration tests
├── core/                         # Backend engine (Modules 1–20)
│   ├── engine.py                 # MedicalAIEngine orchestrator
│   ├── models/                   # Modality specialist predictors
│   ├── xai/                      # GradCAM explainability engine
│   ├── anomaly/                  # OOD detection engine
│   ├── preprocessing/            # Image preprocessing pipelines
│   ├── database.py               # SQLite persistence layer
│   ├── report_generator.py       # Medical report generator
│   └── benchmark.py             # Model evaluation framework
├── app.py                        # Gradio web dashboard
├── config.py                     # Configuration management
├── deployment/                   # Docker, K8s, startup scripts
│   ├── Dockerfile                # Multi-stage build
│   ├── docker-compose.yml        # Service orchestration
│   ├── startup.sh / startup.bat  # Container entrypoints
│   └── kubernetes/               # K8s manifests
├── docs/                         # Documentation
├── paper/                        # IEEE paper assets
├── diagrams/                     # Architecture diagrams (Mermaid)
├── tests/                        # Test suite
├── requirements.txt              # Python dependencies
├── Makefile                      # Developer commands
└── PROJECT_STATE.md              # Module completion tracker
```

---

## 3. Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Docker Desktop (for containerized development)

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd "Medical Image Diagnosis and Anomaly Detection"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Launch development server (all services)
python app.py       # Gradio dashboard → localhost:7860
python -m uvicorn api.main:app --reload  # REST API → localhost:8000
```

---

## 4. Core Components

### 4.1 MedicalAIEngine (`core/engine.py`)

The central orchestrator. **All prediction calls must go through this class.**

```python
from core.engine import MedicalAIEngine

engine = MedicalAIEngine()
result = engine.predict(image_data, modality="brain_mri", patient_id="PAT-001")
```

**Key methods:**
| Method | Description |
|---|---|
| `predict(image, modality, patient_id)` | Run full 8-stage pipeline |
| `batch_predict(images, modality)` | Batch inference |
| `get_supported_modalities()` | List available modalities |
| `get_engine_status()` | Health and status information |

### 4.2 Modality Registry

Each modality is registered in the engine's modality registry mapping modality name → predictor class.

```python
MODALITY_REGISTRY = {
    "brain_mri": BrainMRIPredictor,
    "chest_xray": ChestXRayPredictor,
    "retina": RetinaPredictor,
    # ...
}
```

### 4.3 GradCAM Engine (`core/xai/`)

```python
from core.xai.gradcam import GradCAMEngine

cam_engine = GradCAMEngine(model, target_layer="layer4")
heatmap_b64 = cam_engine.generate(image_tensor, predicted_class_idx)
```

### 4.4 OOD Anomaly Engine (`core/anomaly/`)

```python
from core.anomaly.detector import AnomalyDetector

detector = AnomalyDetector(strategy="mahalanobis")
result = detector.compute_anomaly_score(features, predicted_class)
# result.score, result.risk_level, result.is_anomaly
```

---

## 5. Adding a New Modality Predictor

To add a new imaging modality:

**Step 1:** Create predictor class in `core/models/`:

```python
# core/models/ocular_oct_predictor.py
from core.models.base_predictor import BaseModalityPredictor

class OcularOCTPredictor(BaseModalityPredictor):
    MODALITY_NAME = "ocular_oct"
    CLASS_LABELS = ["Normal", "Drusen", "CNV", "DME"]
    BACKBONE = "efficientnet_b0"

    def predict(self, image_tensor):
        # Implementation here
        return PredictionResult(...)
```

**Step 2:** Register in `core/engine.py`:

```python
from core.models.ocular_oct_predictor import OcularOCTPredictor

MODALITY_REGISTRY["ocular_oct"] = OcularOCTPredictor
```

**Step 3:** Add preprocessing pipeline in `core/preprocessing/` if needed.

**Step 4:** Add test cases in `tests/`.

---

## 6. REST API Internals

### Route Handler Pattern

All route handlers are thin wrappers:

```python
# api/routes/prediction.py
@router.post("/predict")
async def predict(
    file: UploadFile,
    modality: str,
    patient_id: Optional[str] = None,
    engine: MedicalAIEngine = Depends(get_engine),
    _: bool = Depends(verify_api_key)
):
    image_data = await file.read()
    result = engine.predict(image_data, modality, patient_id)
    return result
```

### Middleware Execution Order

Requests pass through middleware in this order:
1. `RequestTracingMiddleware` — injects `X-Request-ID` UUID
2. `ProcessTimingAndLoggingMiddleware` — logs and times requests
3. `SecurityHeadersMiddleware` — adds HSTS, CSP, X-Frame headers
4. `RateLimiterMiddleware` — enforces per-endpoint rate limits
5. `CORSMiddleware` — validates Origin header
6. `verify_api_key` dependency — authenticates API key

### Schema Validation

All request/response bodies are validated by Pydantic schemas in `api/schemas.py`.

---

## 7. Database Schema

The SQLite database (`data/predictions.db`) contains two tables:

```sql
CREATE TABLE prediction_records (
    prediction_id     TEXT PRIMARY KEY,
    patient_id        TEXT,
    modality          TEXT NOT NULL,
    predicted_class   TEXT NOT NULL,
    confidence_score  REAL NOT NULL,
    risk_level        TEXT NOT NULL,
    anomaly_score     REAL,
    is_anomaly        INTEGER DEFAULT 0,
    gradcam_base64    TEXT,
    timestamp         TEXT NOT NULL,
    processing_time_ms REAL,
    report_path       TEXT
);

CREATE TABLE benchmark_results (
    id                TEXT PRIMARY KEY,
    run_timestamp     TEXT NOT NULL,
    modality          TEXT NOT NULL,
    model_name        TEXT,
    accuracy          REAL,
    precision         REAL,
    recall            REAL,
    f1_score          REAL,
    auc_roc           REAL,
    inference_ms      REAL
);
```

---

## 8. Configuration System

Configuration is managed via `config.py` and environment variables. Never hardcode secrets.

```python
from config import get_config

cfg = get_config()
cfg.API_PORT        # int, default 8000
cfg.AUTH_REQUIRED   # bool, default True
cfg.MAX_FILE_SIZE   # int bytes, default 20MB
cfg.LOG_LEVEL       # str, default INFO
```

All configuration keys are documented in `.env.example`.

---

## 9. Testing

```bash
# Run all tests
make test

# Run API tests only
pytest api/tests/ -v

# Run core engine tests
pytest tests/ -v

# Generate coverage report
pytest --cov=core --cov=api --cov-report=html
```

---

## 10. Docker & Deployment

### Build & Run

```bash
docker compose --profile production up --build -d
```

### Container Security

The production container enforces:
- Non-root execution (`medai:1001`)
- Read-only root filesystem
- No privilege escalation
- All Linux capabilities dropped

### Environment Variables

See `.env.example` for all supported variables. Critical variables:

| Variable | Description | Default |
|---|---|---|
| `API_PORT` | FastAPI port | `8000` |
| `GRADIO_PORT` | Gradio port | `7860` |
| `AUTH_REQUIRED` | Enable API key auth | `true` |
| `MEDAI_API_KEY_1` | Primary API key | _(none)_ |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## 11. Code Style & Contribution Guidelines

- **Python**: Follow PEP 8. Use type hints on all function signatures.
- **Docstrings**: Google-style docstrings on all public methods.
- **Imports**: Absolute imports only. No circular imports.
- **Tests**: All new code must be accompanied by test cases.
- **No business logic in API routes**: Logic belongs in `core/engine.py` or domain classes.
- **Architecture is locked**: Do not modify completed modules unless required for integration.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full contribution workflow.

---

## 12. Module Architecture Reference

| Module | Name | Path |
|---|---|---|
| 1–5 | Core Engine & Models | `core/` |
| 6–8 | Preprocessing & Augmentation | `core/preprocessing/` |
| 9–12 | GradCAM & XAI | `core/xai/` |
| 13–15 | Anomaly Detection | `core/anomaly/` |
| 16–18 | Database & Reports | `core/database.py`, `core/report_generator.py` |
| 19–20 | Gradio UI | `app.py` |
| 21 | FastAPI REST API | `api/` |
| 22 | Docker & Deployment | `deployment/` |
| 23 | Documentation | `docs/`, `paper/`, `diagrams/` |
