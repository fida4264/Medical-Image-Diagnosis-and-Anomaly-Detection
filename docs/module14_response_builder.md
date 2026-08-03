# Module 14.2.5: Unified Response Builder & Clinical Report Aggregator Specification

> **System Version:** `0.14.2.5`  
> **Module Status:** COMPLETE & VERIFIED  
> **Component:** `medical_ai_engine/response/`  
> **Author:** DeepMind AI Engineering  
> **Date:** August 2, 2026  

---

## 1. Overview & Architecture

Module 14.2.5 introduces the **Unified Response Builder & Clinical Report Aggregator** subsystem. It is responsible for constructing the final, standardized, immutable prediction response (`PredictionResult`) returned by the `MedicalAIEngine` after execution of the 9-stage prediction pipeline.

```
       EngineContext (Completed)
                  │
                  ▼
       ┌──────────────────────┐
       │   ResponseBuilder    │
       └──────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ ClinicalReport│   │ Response      │
│ Aggregator    │   │ Validator     │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        PredictionResult (frozen=True)
                  │
                  ▼
       ┌──────────────────────┐
       │ ResponseSerializer   │
       │ (JSON / Pretty / Dict)│
       └──────────────────────┘
```

---

## 2. Component Design

### 2.1 Immutable `PredictionResult` (`prediction_result.py`)
Constructed as a `@dataclass(frozen=True)` instance. Once instantiated, no field may be mutated.

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique request UUID4 |
| `session_id` | `str` | Execution session UUID4 |
| `timestamp` | `str` | ISO 8601 UTC timestamp of creation |
| `schema_version` | `str` | Response schema version (`1.0.0`) |
| `status` | `PredictionStatus` | `SUCCESS`, `COMPLETED_WITH_WARNINGS`, `FAILED`, etc. |
| `modality` | `str` | Detected modality string (`chest_xray`, `skin`, etc.) |
| `predicted_disease` | `str` | Diagnosed condition |
| `confidence` | `float` | Primary confidence score (0.0–1.0) |
| `prediction_reliability` | `PredictionReliability` | `high`, `moderate`, `low`, `unknown` |
| `requires_manual_review` | `bool` | Clinician review required flag |
| `routing_decision` | `Dict[str, Any]` | Modality router decision metadata |
| `clinical_report` | `Dict[str, Any]` | Aggregated clinical diagnostic report |
| `performance_metrics` | `Dict[str, float]` | Per-stage timing metrics in ms |
| `model_metadata` | `Dict[str, Any]` | Model name, version, backend |
| `warnings` | `List[str]` | Non-fatal pipeline warnings |
| `errors` | `List[Dict[str, str]]` | Recorded non-fatal pipeline errors |

---

### 2.2 Response Builder (`response_builder.py`)
Implements `IResponseBuilder`. Reads all available signals from `EngineContext` and dispatches to `ClinicalReportAggregator` and `ResponseValidator` before freezing into `PredictionResult`.

- **Primary Path (`build(ctx)`):** Converts context to successful or warning `PredictionResult`.
- **Error Path (`build_error_response(ctx, message)`):** Constructs a fallback `PredictionResult` with `FAILED` status and clinician review flag enabled.

---

### 2.3 Clinical Report Aggregator (`clinical_report_aggregator.py`)
Modality-agnostic aggregator that combines:
- Primary diagnosis & confidence score
- Top-K differential diagnoses
- Auto-generated clinical note
- Clinical recommendations derived from reliability and review flags
- Severity grade (when provided by specialized adapters like retina Module 11)

---

### 2.4 Response Validator (`response_validator.py`)
Implements `IResponseValidator`. Enforces:
1. Presence of non-empty `request_id`, `session_id`, `status`.
2. Confidence within range `[0.0, 1.0]`.
3. Non-empty `predicted_disease` and `modality` for success statuses.

---

### 2.5 Response Serializer (`response_serializer.py`)
Converts `PredictionResult` to:
- `json` (compact JSON string)
- `pretty_json` (indented, formatted JSON string)
- `dict` (Python dictionary)

*Reserved format `fhir_r4` raises `ResponseSerializationException` for future expansion.*

---

## 3. Exceptions (`response_exceptions.py`)

| Class | Error Code | Raised When |
|---|---|---|
| `ResponseException` | `ENG-330` | Base class for response subsystem errors |
| `ResponseBuildException` | `ENG-331` | `ResponseBuilder` encounters unrecoverable context gaps |
| `ResponseValidationException` | `ENG-332` | `PredictionResult` fails validation checks |
| `ResponseSerializationException` | `ENG-333` | Requested format is unsupported or serialization fails |

---

## 4. Verification & Testing

Module 14.2.5 was verified via:
1. `python run_response_builder_tests.py`: 40/40 assertions passed.
2. `python -m pytest medical_ai_engine/tests/test_response_builder.py`: 29/29 PyTest tests passed.
3. 20-thread concurrent response construction test verified thread safety and immutability.
