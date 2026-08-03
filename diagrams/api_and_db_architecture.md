# API, Database & Deployment Architecture Diagrams

---

## 1. REST API Architecture & Middleware Stack

```mermaid
graph TD
    subgraph Client["Clients"]
        HTTP["HTTP Clients (curl, httpx, browser)"]
    end

    subgraph API["FastAPI Application Layer"]
        direction TB
        MW1["RequestTracingMiddleware\n(UUID X-Request-ID injection)"]
        MW2["ProcessTimingAndLoggingMiddleware\n(sanitized request/response logs)"]
        MW3["SecurityHeadersMiddleware\n(HSTS, CSP, X-Frame, nosniff)"]
        MW4["RateLimiterMiddleware\n(20/min predict, 60/min history)"]
        MW5["CORSMiddleware (origin whitelist)"]
        AUTH["verify_api_key (secrets.compare_digest)"]

        subgraph Routers["Route Handlers"]
            R1["/predict  POST"]
            R2["/batch-predict  POST"]
            R3["/history  GET"]
            R4["/reports/{id}  GET"]
            R5["/benchmark  GET"]
            R6["/health/* GET"]
            R7["/system/info  GET"]
        end
    end

    HTTP --> MW1 --> MW2 --> MW3 --> MW4 --> MW5 --> AUTH --> Routers
    R1 & R2 --> ENGINE["MedicalAIEngine Core"]
    R3 & R4 --> DB["SQLite DB"]
    R5 --> BENCH["Benchmark Engine"]
    R6 --> HEALTH["Liveness / Readiness Probes"]
```

---

## 2. SQLite Database Schema (Entity Relationship Diagram)

```mermaid
erDiagram
    PREDICTION_RECORDS {
        TEXT prediction_id PK
        TEXT patient_id
        TEXT modality
        TEXT predicted_class
        REAL confidence_score
        TEXT risk_level
        TEXT anomaly_score
        INTEGER is_anomaly
        TEXT gradcam_base64
        TEXT timestamp
        TEXT processing_time_ms
        TEXT report_path
    }

    BENCHMARK_RESULTS {
        TEXT id PK
        TEXT run_timestamp
        TEXT modality
        TEXT model_name
        REAL accuracy
        REAL precision
        REAL recall
        REAL f1_score
        REAL auc_roc
        REAL inference_ms
    }

    PREDICTION_RECORDS ||--o{ BENCHMARK_RESULTS : "generates"
```

---

## 3. Docker Multi-Stage Build Diagram

```mermaid
graph LR
    subgraph Stage1 ["Stage 1: Builder (python:3.11-slim)"]
        GCC["Install build-essential\ngcc, libgl, libsm"] --> VENV["Create /opt/venv\nvirtualenv"]
        VENV --> DEPS["pip install:\npytorch (cpu), fastapi,\ngradio, pillow, opencv..."]
    end

    subgraph Stage2 ["Stage 2: Runtime (python:3.11-slim)"]
        COPY_VENV["COPY /opt/venv\nfrom builder"] --> LABELS["OCI Image Labels\n(v0.22.1.0, MIT, etc.)"]
        LABELS --> USER["Create non-root user\nmedai:1001"]
        USER --> SRC["COPY application\nsource code"]
        SRC --> DIRS["mkdir /app/data\n/app/logs /app/outputs"]
        DIRS --> EXPOSE["EXPOSE 8000 7860"]
        EXPOSE --> HEALTH["HEALTHCHECK\n/health/liveness"]
        HEALTH --> ENTRY["ENTRYPOINT:\ndeployment/startup.sh"]
    end

    Stage1 --> Stage2
```

---

## 4. Docker Compose Service Topology

```mermaid
graph TD
    subgraph Host["Host Machine"]
        ENVFILE[".env file\n(API_PORT, AUTH_REQUIRED, CPU_LIMIT...)"]
        subgraph Compose["docker-compose.yml — medai_network (bridge)"]
            SVC["medical-ai-platform container\n(profile: production)\nUSER medai:1001\nread_only: true\ncap_drop: ALL"]
            VDATA[("Volume: medai_data\n/app/data")]
            VLOGS[("Volume: medai_logs\n/app/logs")]
            VOUT[("Volume: medai_outputs\n/app/outputs")]
        end
        P1["Port 8000 → FastAPI REST API"]
        P2["Port 7860 → Gradio Dashboard"]
    end

    ENVFILE --> SVC
    SVC --> VDATA & VLOGS & VOUT
    SVC --> P1 & P2
```
