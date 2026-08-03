# Prediction Pipeline & Request Lifecycle Diagrams

---

## 1. Prediction Pipeline — Sequence Diagram

```mermaid
sequenceDiagram
    participant C as Client / Gradio UI
    participant A as FastAPI Router
    participant MW as Middleware Stack
    participant E as MedicalAIEngine
    participant MC as Modality Classifier
    participant P as Disease Predictor
    participant XAI as GradCAM Engine
    participant OOD as Anomaly Engine
    participant RG as Report Generator
    participant DB as SQLite Database

    C->>A: POST /predict (image, modality, patient_id)
    A->>MW: Rate Limit + Auth + Trace Request
    MW->>E: execute_pipeline(request)
    E->>E: Stage 1 — Validate input image & MIME type
    E->>E: Stage 2 — Preprocess (resize, normalize, tensor)
    E->>MC: Stage 3 — Route to correct modality predictor
    MC-->>E: Modality confirmed
    E->>P: Stage 4 — Run inference → class + confidence
    P-->>E: PredictionResult (class, scores, features)
    E->>XAI: Stage 5 — Generate GradCAM heatmap overlay
    XAI-->>E: Base64 GradCAM saliency map
    E->>OOD: Stage 6 — Compute anomaly score (Mahalanobis/AE)
    OOD-->>E: AnomalyResult (score, risk_level, is_anomaly)
    E->>RG: Stage 7 — Persist to DB & generate report
    RG-->>E: Report saved (JSON/MD/PDF)
    E->>DB: Save PredictionRecord
    DB-->>E: record_id
    E->>E: Stage 8 — Assemble ClinicalResponse
    E-->>A: ClinicalPredictionResponse
    A-->>C: HTTP 200 JSON response
```

---

## 2. Prediction Request Lifecycle — State Machine

```mermaid
stateDiagram-v2
    [*] --> Created : Engine instantiated
    Created --> Initialized : load_config() complete
    Initialized --> Running : execute_pipeline() called
    Running --> StageValidation : Stage 1
    StageValidation --> StagePreprocessing : Stage 2
    StagePreprocessing --> StageRouting : Stage 3
    StageRouting --> StagePrediction : Stage 4
    StagePrediction --> StageXAI : Stage 5
    StageXAI --> StageAnomaly : Stage 6
    StageAnomaly --> StagePersistence : Stage 7
    StagePersistence --> StageResponse : Stage 8
    StageResponse --> Completed : response returned
    Completed --> Initialized : ready for next request
    Running --> Failed : exception raised
    Failed --> Initialized : error handled
```

---

## 3. Prediction Workflow Flowchart

```mermaid
flowchart TD
    START(["User uploads medical image"]) --> VALID{"Is file valid?\n(MIME, size, format)"}
    VALID -->|No| ERROR(["Return HTTP 400 INVALID_IMAGE"])
    VALID -->|Yes| PREPROCESS["Preprocess image\n(Resize / Normalize / Tensor)"]
    PREPROCESS --> MODALITY{"Modality\nspecified?"}
    MODALITY -->|No| AUTO["Auto-classify modality\nvia Modality Classifier"]
    MODALITY -->|Yes| ROUTE["Route to specialist predictor"]
    AUTO --> ROUTE
    ROUTE --> PREDICT["Run CNN/ResNet inference\n→ class label + confidence scores"]
    PREDICT --> GRADCAM["Generate GradCAM\nheatmap overlay"]
    GRADCAM --> OOD["Compute OOD anomaly score\n(Mahalanobis / Autoencoder)"]
    OOD --> RISK["Calculate clinical risk level\n(LOW / MEDIUM / HIGH / CRITICAL)"]
    RISK --> SAVE["Persist to SQLite DB\nGenerate Report"]
    SAVE --> RESPOND(["Return Clinical Prediction Response"])
```
