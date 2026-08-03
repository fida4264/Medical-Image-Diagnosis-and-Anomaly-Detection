# Technical Documentation
## Medical Image Diagnosis and Anomaly Detection Platform
### Version 0.22.1.0

---

## 1. Platform Overview

The **Medical Image Diagnosis and Anomaly Detection Platform** is a full-stack, production-grade AI system for automated clinical image analysis. It processes seven distinct medical imaging modalities through an eight-stage inference pipeline and produces clinical-grade predictions with visual explanations, anomaly scores, and structured medical reports.

### Core Capabilities

| Capability | Technology |
|---|---|
| Multi-modality classification | PyTorch CNN (ResNet/EfficientNet) |
| Explainability | Gradient-weighted Class Activation Mapping (GradCAM) |
| Anomaly Detection | Mahalanobis Distance + Autoencoder + Confidence Threshold |
| REST API | FastAPI (ASGI / uvicorn) |
| Web Dashboard | Gradio |
| Report Generation | JSON + Markdown + PDF (reportlab) |
| Database | SQLite (via Python sqlite3) |
| Containerization | Docker + Docker Compose + Kubernetes |

---

## 2. Supported Modalities & Disease Classes

| Modality | Conditions Detected |
|---|---|
| **Brain MRI** | No Tumor, Glioma, Meningioma, Pituitary Tumor |
| **Brain CT** | Normal, Hemorrhage, Ischemic Stroke, Contusion |
| **Chest X-Ray** | Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19, Tuberculosis |
| **Retina** | No DR, Mild DR, Moderate DR, Severe DR, Proliferative DR |
| **Histopathology** | Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, SCLC, Normal, Adenosquamous, Sarcomatoid, Mesothelioma |
| **Skin Lesion** | Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, Vascular Lesion, Melanocytic Nevi |
| **Ultrasound** | Benign, Malignant, Normal |

---

## 3. Eight-Stage Inference Pipeline

| Stage | Name | Description |
|---|---|---|
| 1 | **Input Validation** | MIME type check, file size limit (20 MB), minimum resolution (32×32) |
| 2 | **Preprocessing** | Resize to 224×224, normalize with ImageNet μ/σ, convert to tensor |
| 3 | **Modality Routing** | Map modality string → specialist predictor class |
| 4 | **Disease Prediction** | CNN forward pass → class label + softmax confidence scores |
| 5 | **GradCAM Generation** | Backprop gradient-weighted activation maps → Base64 heatmap overlay |
| 6 | **Anomaly Detection** | Mahalanobis/AE/confidence strategy → normalized anomaly score |
| 7 | **Persistence** | SQLite record insert + medical report file generation |
| 8 | **Response Assembly** | Construct `ClinicalPredictionResponse` with all result fields |

---

## 4. REST API Reference

**Base URL:** `http://localhost:8000`

**Authentication:** API key via `X-API-Key` header (when `AUTH_REQUIRED=true`).

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Single-image prediction |
| `POST` | `/batch-predict` | Multi-image batch prediction |
| `GET` | `/history` | Paginated prediction history |
| `GET` | `/history/{id}` | Single prediction record |
| `GET` | `/reports/{id}` | Download report (JSON/MD/PDF) |
| `GET` | `/benchmark` | Model performance metrics |
| `POST` | `/benchmark/run` | Trigger benchmark run |
| `GET` | `/health/liveness` | Kubernetes liveness probe |
| `GET` | `/health/readiness` | Kubernetes readiness probe |
| `GET` | `/system/info` | Platform version and status |
| `GET` | `/docs` | Swagger UI |

### POST /predict — Request

```
Content-Type: multipart/form-data

file:       (binary) medical image file
modality:   (string) one of: brain_mri, brain_ct, chest_xray, retina,
                     histopathology, skin_lesion, ultrasound
patient_id: (string, optional) patient identifier for audit trail
```

### POST /predict — Response

```json
{
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "patient_id": "PAT-001",
  "modality": "brain_mri",
  "predicted_class": "Glioma",
  "confidence_score": 0.94,
  "all_class_scores": {
    "Glioma": 0.94,
    "No Tumor": 0.04,
    "Meningioma": 0.01,
    "Pituitary": 0.01
  },
  "gradcam_heatmap_b64": "data:image/png;base64,...",
  "anomaly_score": 0.12,
  "risk_level": "LOW",
  "is_anomaly": false,
  "report_path": "outputs/report_550e8400.json",
  "processing_time_ms": 127.4,
  "timestamp": "2026-08-03T13:00:00Z"
}
```

---

## 5. Rate Limits

| Endpoint Group | Rate Limit | Window |
|---|---|---|
| `/predict`, `/batch-predict` | 20 requests | 60 seconds |
| `/history`, `/reports` | 60 requests | 60 seconds |
| `/benchmark` | 10 requests | 60 seconds |
| `/health/*` | Unrestricted | — |

---

## 6. Security Architecture

| Layer | Mechanism |
|---|---|
| Authentication | API key via `secrets.compare_digest()` (timing-safe) |
| Transport | HTTPS (TLS termination at reverse proxy) |
| Headers | HSTS, X-Content-Type-Options, X-Frame-Options, CSP |
| Rate Limiting | Per-IP in-memory sliding window |
| Container | Non-root (medai:1001), read-only FS, cap_drop ALL |
| Secrets | Environment variables only; never hardcoded |

---

## 7. Configuration Reference

All settings controlled via environment variables. See `.env.example`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `API_PORT` | int | `8000` | FastAPI server port |
| `GRADIO_PORT` | int | `7860` | Gradio dashboard port |
| `AUTH_REQUIRED` | bool | `true` | Enable API key authentication |
| `MEDAI_API_KEY_1` | str | — | Primary API key |
| `MEDAI_API_KEY_2` | str | — | Secondary API key (rotation) |
| `RATE_LIMIT_PREDICT` | int | `20` | Predict rate limit per minute |
| `MAX_FILE_SIZE_MB` | int | `20` | Max upload size in MB |
| `LOG_LEVEL` | str | `INFO` | Logging verbosity |
| `DB_PATH` | str | `data/predictions.db` | SQLite database path |
| `REPORTS_DIR` | str | `outputs/` | Medical report output directory |

---

## 8. Performance Benchmarks

| Metric | Value |
|---|---|
| Single image inference | ~100–130 ms |
| GradCAM generation overhead | <5% additional latency |
| OOD detection overhead | <10% additional latency |
| API P95 latency (`/predict`) | ~180 ms |
| SQLite write throughput | ~1,000 records/second |
| Docker image size (runtime) | ~1.2 GB |
| Container startup time | ~8–12 seconds |

---

## 9. Data Flow Diagram

```
[Client] → [FastAPI] → [MedicalAIEngine]
                                │
              ┌─────────────────┼─────────────────────┐
              ▼                 ▼                     ▼
    [Modality Predictor]  [GradCAM Engine]   [Anomaly Detector]
              │                 │                     │
              └─────────────────┼─────────────────────┘
                                ▼
                    [Report Generator]
                                │
                    [SQLite Database]
                                │
                    [ClinicalPredictionResponse]
                                │
                           [Client]
```
