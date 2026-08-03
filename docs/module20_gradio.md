# Module 20 — Professional Gradio Web Application Documentation

**Current Version:** `v0.20.0.0`  
**Status:** ✅ Completed & Verified  

---

## 1. Overview & Objectives

Module 20 provides the **final production-grade user interface** for the Medical Image Diagnosis and Anomaly Detection Platform. Built using Gradio 6.x and customized with a bespoke **Dark Navy Medical Dashboard CSS Theme**, it serves as an enterprise-level clinical decision support interface.

### Key Objectives Achieved:
- **Enterprise Clinical Aesthetic**: Custom CSS dark theme (`#0a0f1e` base, `#0d1b2e` cards, `#0ea5e9` accent) replacing standard Gradio styling.
- **Strict Decoupling**: UI layer operates purely as a presentation layer consuming the underlying `MedicalAIEngine` orchestrator via `backend_bridge.py`.
- **Complete Pipeline Integration**: Supports single-click execution of Modality Router, Disease Classifier, GradCAM Heatmap, Mahalanobis OOD Anomaly Engine, and Report Generator.
- **Multi-Tab Dashboard**:
  1. **🏠 Overview & Dashboard**: Stat metrics cards, supported modalities grid, and engine health status.
  2. **🔬 Diagnostic Analysis**: Dual-column layout for image upload, modality override, diagnosis breakdown, GradCAM side-by-side view, anomaly risk gauge, and report export.
  3. **🗄️ Prediction History**: Searchable, modality-filterable audit log powered by Module 19 SQLite database.
  4. **📊 Global Benchmark**: Performance leaderboard comparing all 7 imaging modalities.
  5. **ℹ️ About Architecture**: Technical summary, core design principles, tech stack tags, and developer credits.

---

## 2. Directory Structure

```
D:/Medical Image Diagnosis and Anomaly Detection/
├── app.py                      # Root Gradio application entry point
├── ui/
│   ├── __init__.py             # UI package initializer
│   ├── ui_configuration.py     # Constants, modality maps, risk badges
│   ├── theme.py                # Dark Navy CSS theme & styles
│   ├── backend_bridge.py       # Engine singleton & backend glue
│   └── components/
│       ├── header.py           # Sticky top navigation banner
│       ├── home_dashboard.py   # Overview statistics & modality grid
│       ├── prediction_tab.py   # Diagnostic analysis & inference flow
│       ├── history_tab.py      # SQLite database query table
│       ├── benchmark_tab.py    # Modality performance leaderboard
│       └── about_tab.py        # System architecture & credits
└── tests/
    └── test_ui.py              # Automated test suite for UI layer
```

---

## 3. How to Launch the Application

Run the application directly using Python:

```bash
python app.py
```

The web server will launch at:
`http://127.0.0.1:7860`

---

## 4. Verification & Testing

The UI test suite verifies component instantiation, backend bridge communication, configuration mappings, and report export.

Run unit tests:

```bash
python -m unittest tests/test_ui.py
```

**Results:** All 6 tests passing.
