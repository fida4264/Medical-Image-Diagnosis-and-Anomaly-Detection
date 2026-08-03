# System Workflows & Flowcharts

---

## 1. End-to-End System Workflow

```mermaid
flowchart TD
    START(["Clinician / Researcher\nOpens Browser"]) --> LAUNCH["Launch Platform\nrun.bat / run.sh / docker compose up"]
    LAUNCH --> GRADIO["Gradio Web Dashboard\nhttp://localhost:7860"]
    GRADIO --> SELECT["Select Medical Image\n& Modality"]
    SELECT --> UPLOAD["Upload DICOM / PNG / JPEG\nvia Gradio or REST API"]
    UPLOAD --> ENGINE["MedicalAIEngine\n8-Stage Pipeline"]
    ENGINE --> RESULT["Clinical Prediction Result\n(Class + Confidence + Risk Level)"]
    RESULT --> XAI_VIEW["View GradCAM\nHeatmap Overlay"]
    RESULT --> OOD_VIEW["View Anomaly Score\n& OOD Risk Assessment"]
    RESULT --> REPORT["Download Medical Report\n(PDF / Markdown / JSON)"]
    RESULT --> HISTORY["View Prediction History\n& Audit Trail"]
    HISTORY --> BENCH["Run Model Benchmark\nComparison Dashboard"]
```

---

## 2. Explainability (GradCAM XAI) Workflow

```mermaid
flowchart TD
    PRED_DONE["Prediction Complete"] --> FEAT["Extract Feature Maps\nfrom CNN Backbone Layer"]
    FEAT --> GRAD["Backpropagate Gradients\nfor Predicted Class"]
    GRAD --> WEIGHT["Compute Global Average\nPooling Weights (α_k)"]
    WEIGHT --> HEATMAP["Generate Weighted Feature\nActivation Map L_GradCAM"]
    HEATMAP --> RELU["Apply ReLU Activation\n(Positive Regions Only)"]
    RELU --> RESIZE["Resize to Input\nImage Dimensions"]
    RESIZE --> NORM["Normalize Heatmap\nto [0, 1] Range"]
    NORM --> OVERLAY["Apply Jet Colormap\nOverlay on Original Image"]
    OVERLAY --> B64["Encode Overlay to\nBase64 PNG String"]
    B64 --> RESP(["Return GradCAM Map\nin API Response"])
```

---

## 3. Anomaly Detection Workflow

```mermaid
flowchart TD
    FEAT["Extract Penultimate Layer\nFeature Vector"] --> STRATEGY{"Select OOD\nDetection Strategy"}
    STRATEGY -->|Primary| MAHAL["Mahalanobis Distance\nfrom class centroid"]
    STRATEGY -->|Fallback| AE["Autoencoder\nReconstruction Error"]
    STRATEGY -->|Tertiary| CONF["Confidence Threshold\nSoftmax Max < 0.5"]
    MAHAL & AE & CONF --> SCORE["Compute Normalized\nAnomaly Score [0-1]"]
    SCORE --> FUSE["Decision Fusion\n(weighted ensemble)"]
    FUSE --> RISK{"Risk Level\nClassification"}
    RISK -->|Score < 0.25| LOW["LOW RISK\nNormal finding"]
    RISK -->|0.25 – 0.50| MEDIUM["MEDIUM RISK\nMonitor & review"]
    RISK -->|0.50 – 0.75| HIGH["HIGH RISK\nClinician review required"]
    RISK -->|Score ≥ 0.75| CRITICAL["CRITICAL\nUrgent clinical attention"]
```

---

## 4. Medical Report Generation Workflow

```mermaid
flowchart TD
    RESP["Clinical Prediction Response"] --> BUILDER["Report Builder Engine\nAssemble report sections"]
    BUILDER --> HEADER["Patient Header\n(ID, date, modality, risk)"]
    BUILDER --> FINDING["Clinical Findings\n(predicted class + confidence)"]
    BUILDER --> EXPLAIN["Explainability Section\n(GradCAM narrative)"]
    BUILDER --> ANOMALY["Anomaly Report\n(OOD score + risk level)"]
    BUILDER --> RECS["Clinical Recommendations\nbased on risk level"]

    HEADER & FINDING & EXPLAIN & ANOMALY & RECS --> RENDER{"Output\nFormats"}
    RENDER --> JSON["JSON Report\n(structured data)"]
    RENDER --> MD["Markdown Report\n(human-readable)"]
    RENDER --> PDF["PDF Report\n(via reportlab)"]
    JSON & MD & PDF --> SAVE["Save to /outputs/\nPersist to SQLite DB"]
```

---

## 5. Deployment Workflow (Docker)

```mermaid
flowchart TD
    DEV["Developer / Reviewer\nClones Repository"] --> COPY["Copy .env.example → .env\nConfigure ports & API keys"]
    COPY --> COMPOSE["docker compose --profile production\nup --build -d"]
    COMPOSE --> BUILD1["Stage 1: Builder\nInstall Python deps into /opt/venv"]
    BUILD1 --> BUILD2["Stage 2: Runtime\nCopy venv + source code\nCreate medai:1001 non-root user"]
    BUILD2 --> STARTUP["Container starts\ndeployment/startup.sh"]
    STARTUP --> API_PROC["uvicorn api.main:app\nPort 8000"]
    STARTUP --> GRADIO_PROC["python app.py\nPort 7860"]
    API_PROC & GRADIO_PROC --> HEALTH_OK{"Health Check\nPasses?"}
    HEALTH_OK -->|Yes| READY(["Platform READY\nhttp://localhost:7860\nhttp://localhost:8000/docs"])
    HEALTH_OK -->|No| RETRY["Docker retries\n3x with 30s interval"]
    RETRY --> HEALTH_OK
```
