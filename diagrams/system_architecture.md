# Architectural Diagrams — Medical AI Platform (`v0.23.0.0`)

This document provides publication-quality Mermaid architectural diagrams for the Medical Image Diagnosis & Anomaly Detection Platform.

---

## 1. Overall System Architecture

```mermaid
graph TD
    subgraph Clients ["Client Layer"]
        GUI["Gradio Web Dashboard (Port 7860)"]
        REST_CLIENT["HTTP REST Clients / EHR Integration"]
        CLI["CLI & Automated Batch Runners"]
    end

    subgraph API_Layer ["API & Gateway Layer"]
        API_GW["FastAPI REST API (Port 8000)"]
        AUTH["Constant-Time Multi-Key Auth"]
        MW["Security, Rate Limiter & Tracing Middleware"]
    end

    subgraph Engine_Layer ["MedicalAIEngine Core Orchestrator"]
        ORCH["Pipeline Orchestrator (8 Execution Stages)"]
        ROUTER["Modality Classifier & Router"]
        REGISTRY["Model Registry & Predictor Factory"]
    end

    subgraph Predictors ["Specialized Modality AI Predictors"]
        P1["Brain MRI Predictor"]
        P2["Chest X-Ray Predictor"]
        P3["Brain CT Predictor"]
        P4["Histopathology Predictor"]
        P5["Retina Predictor"]
        P6["Skin Lesion Predictor"]
        P7["Ultrasound Predictor"]
    end

    subgraph Intelligence ["Advanced AI & Clinical Intelligence"]
        XAI["GradCAM Explainability Engine"]
        OOD["Out-of-Distribution Anomaly Engine"]
        FUSION["Risk Assessment & Decision Fusion"]
    end

    subgraph Infrastructure ["Storage, Reporting & Audit Layer"]
        DB[(SQLite Database - medical_ai.db)]
        REPORTS["Medical Report Generator (PDF/MD/JSON)"]
        BENCH["Global Benchmark System"]
    end

    GUI --> ORCH
    REST_CLIENT --> API_GW
    CLI --> ORCH
    API_GW --> AUTH --> MW --> ORCH

    ORCH --> ROUTER --> REGISTRY
    REGISTRY --> P1 & P2 & P3 & P4 & P5 & P6 & P7

    P1 & P2 & P3 & P4 & P5 & P6 & P7 --> XAI
    P1 & P2 & P3 & P4 & P5 & P6 & P7 --> OOD
    XAI & OOD --> FUSION

    FUSION --> REPORTS
    FUSION --> DB
    ORCH --> BENCH
```

---

## 2. MedicalAIEngine Core Architecture

```mermaid
graph LR
    subgraph Core ["MedicalAIEngine Core"]
        INIT["Engine Initialization"] --> CONFIG["Load Global Config"]
        CONFIG --> READY["State: INITIALIZED"]
        READY --> EXEC["execute_pipeline(request)"]
    end

    subgraph Stages ["8 Pipeline Execution Stages"]
        EXEC --> S1["Stage 1: Input Validation"]
        S1 --> S2["Stage 2: Preprocessing"]
        S2 --> S3["Stage 3: Modality Routing"]
        S3 --> S4["Stage 4: Disease Prediction"]
        S4 --> S5["Stage 5: GradCAM XAI Map Generation"]
        S5 --> S6["Stage 6: OOD Anomaly Detection"]
        S6 --> S7["Stage 7: Report & History Persistence"]
        S7 --> S8["Stage 8: Unified Clinical Response Assembly"]
    end

    S8 --> RESP["Clinical Prediction Response"]
```
