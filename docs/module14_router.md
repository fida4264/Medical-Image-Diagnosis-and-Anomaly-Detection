# Module 14.2.3: Unified Modality Router

> **Module Version:** `0.14.2.3`  
> **Package Location:** `medical_ai_engine/routing/`  
> **Status:** OPERATIONAL & TESTED  
> **Date:** August 1, 2026  

---

## 1. Purpose

Module 14.2.3 implements the **Unified Modality Router** — the intelligent subsystem responsible for connecting the Modality Classifier (Module 6) to the appropriate disease-specific adapter obtained from the `ModelRegistry` (Module 14.2.2).

The Router does **not** perform disease prediction. Its sole responsibility is to evaluate an incoming `EngineContext`, invoke Module 6, enforce confidence and fallback policies, verify adapter availability in `ModelRegistry`, and produce an immutable `RoutingDecision`.

---

## 2. Architecture Overview

```
MedicalAIEngine Core
        │
        ▼
   ModalityRouter  ──► ModalityPredictor (Module 6)
        │
        ├──► ConfidencePolicy  (evaluates per-modality thresholds in configs/routing.yaml)
        ├──► FallbackPolicy    (checks supported modalities & ModelRegistry registration)
        │
        ▼
  RoutingDecision (ROUTE_TO_MODEL | LOW_CONFIDENCE | UNSUPPORTED_MODALITY | MODEL_NOT_REGISTERED | ROUTING_FAILED)
        │
        ▼
 EngineContext.modality_prediction populated
```

---

## 3. Routing Lifecycle & Flow

```
EngineContext Received
        │
        ├── Manual Override Set? ─── YES ──► Check Registry ──► ROUTE_TO_MODEL (mode="manual_override")
        │
        NO
        ▼
Invoke Module 6 ModalityPredictor
        │
        ▼
Apply RoutingPolicy (in order):
  1. FallbackPolicy.evaluate_modality() ── Unsupported? ──► UNSUPPORTED_MODALITY
  2. ConfidencePolicy.evaluate()        ── < Threshold? ──► LOW_CONFIDENCE
  3. FallbackPolicy.evaluate_adapter()   ── Unregistered?─► MODEL_NOT_REGISTERED
        │
        ▼
All Checks Passed ──► ROUTE_TO_MODEL
        │
        ▼
Populate EngineContext:
  - ctx.modality_prediction dict
  - ctx.timestamps["modality_predicted"]
  - ctx.performance_metrics["routing_ms"]
  - ctx.logs, ctx.warnings, ctx.errors
```

---

## 4. Routing Outcomes & Actions

| Outcome | Trigger Condition | Action / Recommendation |
|---|---|---|
| `ROUTE_TO_MODEL` | Modality supported, confidence ≥ threshold, adapter registered in `ModelRegistry` | `proceed_to_inference` |
| `LOW_CONFIDENCE` | Module 6 confidence < per-modality threshold or global floor | `request_manual_review_or_rescan` |
| `UNSUPPORTED_MODALITY` | Detected modality not in `configs/routing.yaml` `supported_modalities` | `verify_image_modality_and_retry` |
| `MODEL_NOT_REGISTERED` | Modality supported & confidence passed, but no adapter registered in `ModelRegistry` | `register_adapter_and_retry` |
| `ROUTING_FAILED` | Unexpected internal exception or classifier failure | `inspect_logs_and_retry` |

---

## 5. Confidence Policy & Per-Modality Thresholds

Thresholds are stored in `configs/routing.yaml`:

| Modality Key | Acceptance Threshold | Description / Risk Profile |
|---|---|---|
| `brain_mri` | `0.90` | High precision requirement for intracranial lesions |
| `chest_xray` | `0.85` | Multi-label pulmonary screening |
| `brain_ct` | `0.88` | Acute hemorrhage detection |
| `histopathology` | `0.92` | High-risk oncological grading |
| `retina` | `0.88` | Diabetic retinopathy screening |
| `skin` | `0.85` | Lesion classification |
| `ultrasound` | `0.83` | Acoustic image analysis |

- **Global Confidence Floor (`0.60`):** Rejects any prediction below 0.60 regardless of modality threshold.
- **Warning Threshold (`0.70`):** Adds a non-fatal warning to `ctx.warnings` if confidence is between 0.70 and the modality threshold.

---

## 6. Integration Rules

- **Zero Direct Disease Model Imports:** `ModalityRouter` communicates *only* with `ModalityPredictor` (Module 6) and `ModelRegistry` (Module 14.2.2). It does not import or depend on Modules 7–13.
- **Pure Context Interface:** Interacts with the engine strictly through `EngineContext` reading and mutating `ctx.modality_prediction`.
- **Thread-Safety:** Router instance is thread-safe and supports concurrent routing requests across worker threads.

---

## 7. Future Extension Points

- **Dynamic Threshold Reloading:** Can reload `configs/routing.yaml` at runtime without engine restart.
- **Multi-Modality Fusion Routing:** Architecture supports top-K candidate evaluation for composite routing in future releases.
- **Audit Logging & Telemetry:** Full integration ready for Prometheus/Grafana or Database logging stages (Module 19).

---

*Module 14.2.3: Unified Modality Router Specification — v0.14.2.3*
