<div align="center">

# 🏥 Medical Image Diagnosis & Anomaly Detection Platform

### Multi-Modality Deep Learning System for Clinical Decision Support

[![Version](https://img.shields.io/badge/version-0.22.1.0-blue.svg)](https://github.com/your-repo)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](deployment/Dockerfile)
[![Modules](https://img.shields.io/badge/modules-23%2F23-brightgreen.svg)](PROJECT_STATE.md)

**B.Tech Final Year Project — Medical AI Research Platform**

[📖 Documentation](docs/) · [🔬 Paper Assets](paper/) · [📐 Diagrams](diagrams/) · [🚀 Quick Start](#-quick-start) · [🔌 API Reference](docs/API_REFERENCE.md)

</div>

---

## 🔬 Overview

The **Medical Image Diagnosis and Anomaly Detection Platform** is a production-grade, full-stack AI system that unifies seven distinct medical imaging modalities under a single, orchestrated deep learning framework. It delivers:

- 🧠 **Disease Classification** across 40+ clinical conditions
- 🗺️ **GradCAM Visual Explanations** — anatomically localized AI evidence
- ⚠️ **OOD Anomaly Detection** — clinical safety through distribution monitoring
- 📄 **Automated Medical Reports** (PDF, Markdown, JSON)
- 🌐 **Production REST API** with security hardening (FastAPI)
- 💻 **Web Dashboard** (Gradio) for direct clinical use
- 🐳 **Docker + Kubernetes** deployment artifacts

> ⚕️ **Clinical Disclaimer**: This platform is a decision-support research tool. All predictions must be reviewed by a qualified clinician before clinical application.

---

## 🎯 Supported Modalities

| # | Modality | Conditions Detected | Classes |
|---|---|---|---|
| 1 | 🧠 Brain MRI | No Tumor, Glioma, Meningioma, Pituitary | 4 |
| 2 | 🧠 Brain CT | Normal, Hemorrhage, Ischemic Stroke, Contusion | 4 |
| 3 | 🫁 Chest X-Ray | Normal, Pneumonia, COVID-19, Tuberculosis | 4+ |
| 4 | 👁️ Retina | No DR, Mild/Moderate/Severe/Proliferative DR | 5 |
| 5 | 🔬 Histopathology | Adenocarcinoma, Squamous Cell, SCLC + more | 8 |
| 6 | 🩹 Skin Lesion | Melanoma, BCC, SCC, Nevi + more | 9 |
| 7 | 📡 Ultrasound | Benign, Malignant, Normal | 3 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Presentation Layer                              │
│   Gradio Web Dashboard (port 7860) │ FastAPI REST (port 8000)│
├─────────────────────────────────────────────────────────────┤
│              Orchestration Layer                             │
│              MedicalAIEngine — 8-Stage Pipeline              │
├──────────────┬─────────────────────┬────────────────────────┤
│  Prediction  │   Explainability    │   Anomaly Detection    │
│  (7 CNNs)    │   (GradCAM Engine)  │   (OOD Engine)         │
├──────────────┴─────────────────────┴────────────────────────┤
│              Persistence & Reporting Layer                   │
│         SQLite Database │ Medical Report Generator           │
└─────────────────────────────────────────────────────────────┘
```

**Full diagrams:** [diagrams/system_architecture.md](diagrams/system_architecture.md)

---

## 🚀 Quick Start

### Option 1: One-Command Launch (Windows)

```bat
run.bat
```

### Option 2: One-Command Launch (Linux / macOS)

```bash
chmod +x run.sh && ./run.sh
```

### Option 3: Docker (Recommended for Production)

```bash
# Configure environment
cp .env.example .env

# Launch production stack
docker compose --profile production up --build -d
```

| Service | URL |
|---|---|
| 💻 Web Dashboard | http://localhost:7860 |
| 🔌 REST API | http://localhost:8000 |
| 📚 API Docs (Swagger) | http://localhost:8000/docs |

---

## 🔌 API Usage

```bash
# Analyze a medical image
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@brain_mri.jpg" \
  -F "modality=brain_mri" \
  -F "patient_id=PAT-001"
```

**Response:**
```json
{
  "predicted_class": "Glioma",
  "confidence_score": 0.9412,
  "risk_level": "LOW",
  "anomaly_score": 0.124,
  "gradcam_heatmap_b64": "data:image/png;base64,...",
  "processing_time_ms": 127.4
}
```

Full API reference: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

---

## 📁 Repository Structure

```
Medical Image Diagnosis and Anomaly Detection/
├── api/                    # FastAPI REST API (Module 21)
│   ├── main.py             # ASGI app + middleware
│   ├── routes/             # Endpoint handlers
│   ├── schemas.py          # Pydantic models
│   └── middleware.py       # Security, tracing, rate limiting
├── core/                   # AI Engine (Modules 1–20)
│   ├── engine.py           # MedicalAIEngine orchestrator
│   ├── models/             # 7 modality specialist CNNs
│   ├── xai/                # GradCAM explainability
│   ├── anomaly/            # OOD detection engine
│   └── report_generator.py # PDF/MD/JSON medical reports
├── deployment/             # Docker & Kubernetes (Module 22)
│   ├── Dockerfile          # Multi-stage build
│   ├── docker-compose.yml  # Production orchestration
│   └── kubernetes/         # K8s manifests
├── docs/                   # Documentation (Module 23)
│   ├── USER_MANUAL.md
│   ├── DEVELOPER_MANUAL.md
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   └── PROJECT_REPORT.md
├── paper/                  # IEEE Paper Assets (Module 23)
│   ├── ABSTRACT.md
│   ├── PROBLEM_STATEMENT.md
│   ├── LITERATURE_REVIEW.md
│   ├── PROPOSED_METHODOLOGY.md
│   ├── RESULTS_AND_DISCUSSION.md
│   └── REFERENCES_TEMPLATE.md
├── diagrams/               # Architecture Diagrams (Module 23)
│   ├── system_architecture.md
│   ├── prediction_pipeline.md
│   ├── api_and_db_architecture.md
│   └── workflows_and_flowcharts.md
├── app.py                  # Gradio web dashboard
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── Makefile                # Developer commands
├── run.bat                 # Windows quick-launch
├── run.sh                  # Linux/macOS quick-launch
└── PROJECT_STATE.md        # Module completion tracker
```

---

## 🧪 Development

```bash
# Install in development mode
pip install -r requirements.txt

# Run all tests
make test

# Run benchmark
make benchmark

# Start development servers
make dev
```

---

## 📊 Module Completion Status

| Module | Name | Status |
|---|---|---|
| 1–5 | Core Engine & AI Models | ✅ Complete |
| 6–12 | Preprocessing & Modality Models | ✅ Complete |
| 13–14 | GradCAM Explainability | ✅ Complete |
| 15–16 | OOD Anomaly Detection | ✅ Complete |
| 17–18 | Database & Report Generation | ✅ Complete |
| 19–20 | Gradio Web Dashboard | ✅ Complete |
| 21 | FastAPI REST API | ✅ Complete |
| 22 | Docker & Deployment | ✅ Complete |
| 23 | Documentation & Research Assets | ✅ Complete |

**Current Version:** `v0.22.1.0` — All 23 modules complete ✅

---

## 📚 Documentation Index

| Document | Description |
|---|---|
| [User Manual](docs/USER_MANUAL.md) | End-user guide for the platform |
| [Developer Manual](docs/DEVELOPER_MANUAL.md) | Architecture and developer guide |
| [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) | System specs and API reference |
| [API Reference](docs/API_REFERENCE.md) | Full endpoint documentation |
| [Project Report](docs/PROJECT_REPORT.md) | FYP academic submission report |
| [Abstract](paper/ABSTRACT.md) | IEEE paper abstract |
| [Methodology](paper/PROPOSED_METHODOLOGY.md) | Technical methodology document |
| [Literature Review](paper/LITERATURE_REVIEW.md) | Related work survey |
| [System Diagrams](diagrams/system_architecture.md) | Architecture Mermaid diagrams |

---

## 🛡️ Security

This project follows responsible security practices. See [SECURITY.md](SECURITY.md) for the vulnerability reporting policy.

**Production security features:**
- API key authentication (timing-safe comparison)
- Security headers (HSTS, CSP, X-Frame-Options)
- Rate limiting per endpoint
- Non-root container execution
- Read-only container filesystem
- All capabilities dropped

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

---

<div align="center">

**Built with ❤️ for clinical AI research**

*Medical Image Diagnosis and Anomaly Detection Platform — v0.22.1.0*

</div>
