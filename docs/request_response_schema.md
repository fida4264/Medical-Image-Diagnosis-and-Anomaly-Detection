# Request & Response Schema — Medical AI Orchestration Engine (Module 14)

> **Module:** 14 — Unified Medical AI Orchestration Engine  
> **Specification Version:** 1.1  
> **Last Updated:** August 1, 2026  

---

## 1. `UnifiedPredictionRequest` — Input Schema

### Python Dataclass Definition
```python
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np


@dataclass
class UnifiedPredictionRequest:
    """
    Input contract for the MedicalAIOrchestrator.predict() method.
    """
    # ── Required ──────────────────────────────────────────────────
    image_source: Union[str, Path, Image.Image, bytes, np.ndarray]
    """
    The medical image to diagnose. Accepted formats:
        - str / Path   → absolute or relative file path
        - PIL.Image    → already-decoded RGB image
        - bytes        → raw image bytes (JPEG/PNG)
        - np.ndarray   → HxWxC uint8 RGB array
    """

    # ── Optional Controls ─────────────────────────────────────────
    manual_modality_override: Optional[str] = None
    """
    If provided, skips Module 6 modality detection and routes directly
    to the specified modality engine.
    Valid values: "brain_mri" | "chest_xray" | "brain_ct" |
                  "histopathology" | "retina" | "skin" | "ultrasound"
    """

    top_k: int = 3
    """Number of top disease predictions to include in the response (1–7)."""

    inference_backend: str = "pytorch"
    """Inference execution backend: "pytorch" | "onnx" | "torchscript"."""

    enable_clinical_report: bool = True
    """If True, assembles disease-specific clinical diagnostic notes."""

    enable_gradcam: bool = False
    """If True, generates Grad-CAM attention map (requires Module 15)."""

    enable_anomaly_detection: bool = False
    """If True, runs anomaly scorer (requires Module 16)."""

    request_id: Optional[str] = None
    """Optional caller-supplied request identifier. Auto-generated if None."""
```

### JSON Equivalent
```json
{
  "image_source": "/path/to/medical_image.jpg",
  "manual_modality_override": null,
  "top_k": 3,
  "inference_backend": "pytorch",
  "enable_clinical_report": true,
  "enable_gradcam": false,
  "enable_anomaly_detection": false,
  "request_id": null
}
```

---

## 2. `UnifiedPredictionResponse` — Output Schema

### Full JSON Schema (All Fields)
```json
{
  "request_id": "req_20260801_c4f9a1b2",

  "modality_routing": {
    "detected_modality": "skin",
    "modality_confidence": 0.9984,
    "routing_mode": "automatic",
    "routing_time_ms": 4.2,
    "low_confidence_warning": false
  },

  "prediction": {
    "module": "skin",
    "modality": "skin",
    "predicted_class": "melanoma",
    "confidence": 0.9312,

    "prediction_reliability": "high",
    "requires_manual_review": false,

    "severity_grade": null,

    "top_k_predictions": [
      {"class": "melanoma",            "confidence": 0.9312},
      {"class": "melanocytic_nevus",   "confidence": 0.0410},
      {"class": "basal_cell_carcinoma","confidence": 0.0210}
    ],

    "probability_distribution": {
      "melanoma":             0.9312,
      "melanocytic_nevus":    0.0410,
      "basal_cell_carcinoma": 0.0210,
      "benign_keratosis":     0.0030,
      "actinic_keratosis":    0.0020,
      "vascular_lesion":      0.0010,
      "dermatofibroma":       0.0008
    }
  },

  "clinical_report": {
    "report_type": "skin_lesion_cancer_classification",
    "primary_diagnosis": "melanoma",
    "reliability": "high",
    "requires_manual_review": false,
    "differential_diagnoses": ["melanocytic_nevus", "basal_cell_carcinoma"],
    "clinical_note": "HIGH RISK FINDING: Malignant Melanoma predicted with [high] reliability. Urgent dermatological evaluation, dermoscopy, and excisional biopsy recommended."
  },

  "gradcam": null,

  "anomaly": null,

  "execution_metadata": {
    "model_version": "0.14.1",
    "validation_time_ms": 0.5,
    "quality_check_time_ms": 1.1,
    "routing_time_ms": 4.2,
    "inference_time_ms": 11.8,
    "report_assembly_time_ms": 1.0,
    "total_execution_time_ms": 18.6,
    "timestamp": "2026-08-01T20:55:00.123456Z",
    "status": "success",
    "warnings": []
  }
}
```

---

## 3. Field-by-Field Schema Reference

### `modality_routing` Object

| Field | Type | Description |
|---|---|---|
| `detected_modality` | `string` | Lowercase modality key detected or supplied |
| `modality_confidence` | `float` | Module 6 confidence score (0.0 – 1.0) |
| `routing_mode` | `string` | `"automatic"` (Module 6) or `"manual_override"` |
| `routing_time_ms` | `float` | Time taken for modality classification |
| `low_confidence_warning` | `bool` | `true` if confidence < 0.70 |

### `prediction` Object

| Field | Type | Present | Description |
|---|---|---|---|
| `module` | `string` | Always | Source module name |
| `modality` | `string` | Always | Target modality |
| `predicted_class` | `string` | Always | Top-1 disease class name |
| `confidence` | `float` | Always | Top-1 confidence score |
| `prediction_reliability` | `string` | Modules 12, 13 | `"high"` / `"moderate"` / `"low"` |
| `requires_manual_review` | `bool` | Modules 12, 13 | Clinical safety flag |
| `severity_grade` | `int` / `null` | Module 11 | Diabetic retinopathy Grade 0–4 |
| `top_k_predictions` | `array` | Always | Top-k class/confidence pairs |
| `probability_distribution` | `dict` | Always | Full class probability distribution |

### `clinical_report` Object (when `enable_clinical_report: true`)

| Field | Type | Description |
|---|---|---|
| `report_type` | `string` | Module-specific report type identifier |
| `primary_diagnosis` | `string` | Top-1 predicted disease class |
| `reliability` | `string` | Clinical reliability level |
| `requires_manual_review` | `bool` | Clinical safety flag |
| `differential_diagnoses` | `array` | Top-k minus 1 disease alternatives |
| `clinical_note` | `string` | Disease-specific clinical guidance paragraph |

### `gradcam` Object (when `enable_gradcam: true`, Module 15 active)

```json
{
  "heatmap_path": "outputs/orchestration/gradcam/req_001_gradcam.png",
  "target_layer": "features.7",
  "cam_method": "gradcam",
  "generation_time_ms": 42.3
}
```

### `anomaly` Object (when `enable_anomaly_detection: true`, Module 16 active)

```json
{
  "anomaly_score": 0.12,
  "is_anomaly": false,
  "anomaly_threshold": 0.75,
  "ood_confidence": 0.08,
  "detection_time_ms": 15.2
}
```

### `execution_metadata` Object

| Field | Type | Description |
|---|---|---|
| `model_version` | `string` | System version at inference time |
| `validation_time_ms` | `float` | Input validation stage latency |
| `quality_check_time_ms` | `float` | Image quality check stage latency |
| `routing_time_ms` | `float` | Modality detection stage latency |
| `inference_time_ms` | `float` | Disease model forward pass latency |
| `report_assembly_time_ms` | `float` | Clinical report assembly latency |
| `total_execution_time_ms` | `float` | End-to-end pipeline latency |
| `timestamp` | `string` | ISO 8601 UTC timestamp |
| `status` | `string` | `"success"` / `"warning"` / `"partial"` / `"failed"` |
| `warnings` | `array` | List of non-fatal warning strings |

---

## 4. Response `status` Enum

| Value | Meaning |
|---|---|
| `"success"` | All 12 stages completed successfully |
| `"warning"` | Completed with non-fatal warnings (e.g. low routing confidence) |
| `"partial"` | Completed but some optional stages failed (e.g. ONNX fallback used) |
| `"failed"` | Hard failure — exception raised; prediction not available |

---

## 5. Per-Module Schema Compatibility Matrix

| Module | `prediction_reliability` | `requires_manual_review` | `severity_grade` | `gradcam` hook | `anomaly` hook |
|---|---|---|---|---|---|
| Module 7 (Brain MRI) | — | — | — | Planned (Mod 15) | Planned (Mod 16) |
| Module 8 (Chest X-ray) | — | — | — | Planned (Mod 15) | Planned (Mod 16) |
| Module 9 (Brain CT) | — | — | — | Planned (Mod 15) | Planned (Mod 16) |
| Module 10 (Histopathology) | — | — | — | Planned (Mod 15) | Planned (Mod 16) |
| Module 11 (Retina) | — | — | ✔ Grade 0–4 | Planned (Mod 15) | Planned (Mod 16) |
| Module 12 (Skin) | ✔ high/moderate/low | ✔ | — | Planned (Mod 15) | Planned (Mod 16) |
| Module 13 (Ultrasound) | ✔ high/moderate/low | ✔ | — | Planned (Mod 15) | Planned (Mod 16) |

Absent optional fields are serialized as `null` in the JSON response.

---

## 6. Error Response Schema

Returned when `status == "failed"`:

```json
{
  "request_id": "req_20260801_fail_001",
  "status": "failed",
  "error": {
    "stage": "ValidateInputStage",
    "error_type": "ImageValidationError",
    "message": "Cannot decode image: file is corrupt or unsupported format.",
    "supported_formats": ["JPEG", "PNG", "TIFF", "BMP"],
    "timestamp": "2026-08-01T20:55:05Z"
  },
  "prediction": null,
  "clinical_report": null,
  "execution_metadata": {
    "total_execution_time_ms": 1.2,
    "status": "failed"
  }
}
```

---

*Request & Response Schema Reference — Medical AI Orchestration Engine (Module 14)*  
*Version 1.1 — Full Schema Coverage — August 1, 2026*
