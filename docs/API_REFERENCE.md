# API Reference — Complete Endpoint Documentation
## Medical Image Diagnosis and Anomaly Detection Platform
### Version 0.22.1.0

---

## Base URL

```
http://localhost:8000
```

Interactive documentation (Swagger UI): http://localhost:8000/docs
ReDoc documentation: http://localhost:8000/redoc

---

## Authentication

All endpoints (except `/health/*`) require an API key when `AUTH_REQUIRED=true`.

**Header:**
```
X-API-Key: <your_api_key>
```

API keys are configured via environment variables `MEDAI_API_KEY_1`, `MEDAI_API_KEY_2`, etc.

---

## Error Codes

| HTTP Status | Error Code | Description |
|---|---|---|
| 400 | `INVALID_IMAGE` | Unsupported format or corrupted file |
| 400 | `INVALID_MODALITY` | Unrecognized modality string |
| 400 | `FILE_TOO_LARGE` | File exceeds `MAX_FILE_SIZE_MB` limit |
| 401 | `UNAUTHORIZED` | Missing or invalid API key |
| 404 | `NOT_FOUND` | Prediction ID not found |
| 422 | `VALIDATION_ERROR` | Request body schema error |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Engine or infrastructure error |

---

## Endpoints

---

### POST /predict

Run full medical image analysis pipeline on a single image.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | ✅ | Medical image (JPEG/PNG/BMP/TIFF/WEBP, max 20MB) |
| `modality` | string | ✅ | One of: `brain_mri`, `brain_ct`, `chest_xray`, `retina`, `histopathology`, `skin_lesion`, `ultrasound` |
| `patient_id` | string | ❌ | Patient identifier for audit trail |

**Response:** `200 OK`

```json
{
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "patient_id": "PAT-001",
  "modality": "brain_mri",
  "predicted_class": "Glioma",
  "confidence_score": 0.9412,
  "all_class_scores": {
    "Glioma": 0.9412,
    "No Tumor": 0.0401,
    "Meningioma": 0.0127,
    "Pituitary": 0.0060
  },
  "gradcam_heatmap_b64": "data:image/png;base64,iVBORw0KGgo...",
  "anomaly_score": 0.1240,
  "risk_level": "LOW",
  "is_anomaly": false,
  "report_path": "outputs/report_550e8400.json",
  "processing_time_ms": 127.4,
  "timestamp": "2026-08-03T13:00:00Z"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@brain_mri_scan.jpg" \
  -F "modality=brain_mri" \
  -F "patient_id=PAT-001"
```

---

### POST /batch-predict

Run pipeline on multiple images in a single request.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | binary[] | ✅ | List of medical image files |
| `modality` | string | ✅ | Modality applied to all images |

**Response:** `200 OK` — Array of `PredictionResponse` objects.

---

### GET /history

Retrieve paginated prediction history.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `page` | int | `1` | Page number |
| `page_size` | int | `20` | Records per page (max 100) |
| `patient_id` | string | — | Filter by patient ID |
| `modality` | string | — | Filter by modality |
| `start_date` | string | — | ISO 8601 date filter (inclusive) |
| `end_date` | string | — | ISO 8601 date filter (inclusive) |

**Response:** `200 OK`

```json
{
  "total": 142,
  "page": 1,
  "page_size": 20,
  "records": [...]
}
```

---

### GET /history/{prediction_id}

Retrieve a single prediction record by ID.

**Path Parameters:**

| Parameter | Description |
|---|---|
| `prediction_id` | UUID of the prediction |

**Response:** `200 OK` — Single `PredictionResponse` object.

---

### GET /reports/{prediction_id}

Download the medical report for a prediction.

**Query Parameters:**

| Parameter | Type | Options | Default |
|---|---|---|---|
| `format` | string | `json`, `markdown`, `pdf` | `json` |

**Response:** 
- JSON: `application/json`
- Markdown: `text/markdown`
- PDF: `application/pdf` (file download)

---

### GET /benchmark

Retrieve model performance benchmark results.

**Response:** `200 OK`

```json
{
  "benchmark_id": "bm_20260803_130000",
  "run_timestamp": "2026-08-03T13:00:00Z",
  "modality_results": [
    {
      "modality": "brain_mri",
      "accuracy": 0.9234,
      "precision": 0.9156,
      "recall": 0.9087,
      "f1_score": 0.9121,
      "auc_roc": 0.9612,
      "avg_inference_ms": 124.3
    }
  ]
}
```

---

### POST /benchmark/run

Trigger a new benchmark evaluation run. Returns a task ID for polling.

**Response:** `202 Accepted`

```json
{
  "task_id": "task_bm_20260803_130001",
  "status": "queued",
  "message": "Benchmark run started"
}
```

---

### GET /health/liveness

Kubernetes liveness probe. Returns HTTP 200 if the application process is alive.

**Response:** `200 OK`

```json
{"status": "alive", "version": "0.22.1.0"}
```

---

### GET /health/readiness

Kubernetes readiness probe. Returns HTTP 200 only if all components are initialized.

**Response:** `200 OK`

```json
{
  "status": "ready",
  "engine": "initialized",
  "database": "connected",
  "models_loaded": 7
}
```

---

### GET /system/info

Returns platform version, configuration, and runtime information.

**Response:** `200 OK`

```json
{
  "version": "0.22.1.0",
  "environment": "production",
  "python_version": "3.11.x",
  "pytorch_version": "2.x",
  "supported_modalities": ["brain_mri", "brain_ct", "chest_xray", "retina", "histopathology", "skin_lesion", "ultrasound"],
  "auth_enabled": true,
  "rate_limiting_enabled": true,
  "uptime_seconds": 3600
}
```
