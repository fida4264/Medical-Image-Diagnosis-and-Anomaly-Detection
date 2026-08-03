# Module 21 — Production REST API Documentation

## Executive Overview

Module 21 introduces a production-grade, high-performance, hardened **FastAPI REST API** exposing the full capability of the **Medical Image Diagnosis and Anomaly Detection Platform**.

The REST API acts as a thin service layer directly wrapping the underlying, verified backend engines:
- `MedicalAIEngine` (Pipeline orchestration & inference)
- `DatabaseManager` & `PredictionRepository` (SQLite prediction history)
- `ReportGenerator` & `ReportExporter` (Clinical report synthesis and JSON/MD/PDF rendering)
- `BenchmarkRunner` (Platform evaluation metrics across all 7 modalities)

---

## System Architecture

```
[ HTTP Client / Frontend / cURL / Python ]
                       │
                       ▼
         ┌──────────────────────────┐
         │      FastAPI Engine      │
         │      (api/main.py)       │
         └─────────────┬────────────┘
                       │
 ┌─────────────────────┼─────────────────────┐
 ▼                     ▼                     ▼
Security Middleware    Rate Limiter          Tracing Middleware
(HSTS, CSP, X-Frame)   (Sliding Window)      (X-Request-ID)
 │                     │                     │
 └─────────────────────┼─────────────────────┘
                       │
  ┌────────────────────┼────────────────────┐
  ▼                    ▼                    ▼
[/predict]         [/history]           [/reports]
  │                    │                    │
  ▼                    ▼                    ▼
MedicalAIEngine    QueryEngine       ReportExporter
(Singleton)     (SQLite DB WAL)     (PDF/JSON/MD)
```

---

## Production Security & Hardening Features

1. **Multi-Key Constant-Time Authentication**: Uses `secrets.compare_digest` to prevent timing attacks. Supports multiple API keys and key rotation.
2. **Environment-Based Settings**: Managed via `api/settings.py` and `.env` configuration.
3. **HTTP Security Headers**: Enforces HSTS, X-Content-Type-Options (`nosniff`), X-Frame-Options (`DENY`), Referrer-Policy, Permissions-Policy, Content-Security-Policy, and masks `Server` version headers.
4. **Endpoint Rate Limiting**: In-memory token bucket rate limiting per client IP:
   - `/predict`: 20 requests/minute
   - `/history`: 60 requests/minute
   - `/health`: 120 requests/minute
5. **Strict File Upload Sanitization**: 20MB file size limit, MIME type whitelist, extension validation, PIL header decoding.
6. **OpenAPI Docs Protection**: Swagger UI docs can be disabled in production (`ENABLE_DOCS=false`).
7. **Observability Probes**: Dedicated `/health/liveness` and `/health/readiness` Kubernetes probes.
8. **Sanitized Logs & Error Responses**: PHI and API keys masked in logs; stack traces suppressed in production.

---

## Endpoint Reference

### 1. Diagnostic Inference (`/predict` & `/batch-predict`)

#### `POST /predict`
Upload a single medical image for complete diagnostic analysis.

**Headers:**
- `X-API-Key` (string, required if auth enabled): Valid API key.

**Form Parameters:**
- `file` (UploadFile, required): Medical scan image file (max 20MB).
- `modality` (string, optional): Target modality override (`brain_mri`, `chest_xray`, `brain_ct`, `histopathology`, `retina`, `skin`, `ultrasound`).
- `patient_id` (string, optional): Patient / Case ID.

---

#### `POST /batch-predict`
Upload multiple medical scan images for parallel batch processing.

---

### 2. Prediction History (`/history`)

#### `GET /history`
Query, filter, and paginate stored prediction records.

---

### 3. Medical Reports (`/reports`)

#### `GET /reports/{prediction_id}`
Get report metadata and direct file download URLs.

#### `GET /reports/{prediction_id}/download`
Download report in `pdf`, `json`, or `md` format.

---

### 4. System Health & Probes (`/health`)

#### `GET /health/liveness`
Liveness probe returning `200 OK` `{"status": "UP"}`.

#### `GET /health/readiness`
Readiness probe checking engine and database readiness.

#### `GET /health`
Detailed system health telemetry.

---

## Running the Production Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```
