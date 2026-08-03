# User Manual
## Medical Image Diagnosis and Anomaly Detection Platform
### Version 0.22.1.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation & Launch](#3-installation--launch)
4. [Web Dashboard (Gradio)](#4-web-dashboard-gradio)
5. [Running a Prediction](#5-running-a-prediction)
6. [Understanding Results](#6-understanding-results)
7. [Medical Reports](#7-medical-reports)
8. [Prediction History](#8-prediction-history)
9. [Benchmark Dashboard](#9-benchmark-dashboard)
10. [REST API Usage](#10-rest-api-usage)
11. [Troubleshooting](#11-troubleshooting)
12. [Frequently Asked Questions](#12-frequently-asked-questions)

---

## 1. Introduction

The **Medical Image Diagnosis and Anomaly Detection Platform** is an AI-powered clinical decision-support tool that automatically analyzes medical images across seven imaging modalities and provides:

- Disease classification with confidence scores
- Visual explanations via GradCAM saliency maps
- Anomaly (Out-of-Distribution) detection and risk stratification
- Structured medical reports (PDF, Markdown, JSON)
- Prediction audit history

> ⚠️ **Clinical Disclaimer**: This platform is designed as a decision-support tool. All predictions must be reviewed and validated by a qualified clinician before clinical use.

---

## 2. System Requirements

### Minimum Requirements
| Component | Requirement |
|---|---|
| Operating System | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| Python | 3.9 or higher |
| RAM | 8 GB |
| Disk Space | 5 GB |
| Browser | Chrome 90+, Firefox 88+, Edge 90+ |

### Recommended
| Component | Recommendation |
|---|---|
| RAM | 16 GB |
| CPU | 8-core processor |
| Internet | For initial dependency download |

---

## 3. Installation & Launch

### Option A: Quick Start (Windows)

```bat
# Double-click run.bat
# Or run from Command Prompt:
run.bat
```

### Option B: Quick Start (Linux / macOS)

```bash
# Make executable and run:
chmod +x run.sh
./run.sh
```

### Option C: Docker (Recommended for Production)

```bash
# Copy environment template
cp .env.example .env

# Edit API keys (optional)
notepad .env   # Windows
nano .env      # Linux

# Launch
docker compose --profile production up --build -d
```

### After Launch
The platform starts two services automatically:

| Service | URL |
|---|---|
| Web Dashboard | http://localhost:7860 |
| REST API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

---

## 4. Web Dashboard (Gradio)

Open your browser and navigate to **http://localhost:7860**.

You will see the main dashboard with the following panels:

| Panel | Description |
|---|---|
| **🔬 Image Upload** | Upload medical images for analysis |
| **📊 Results** | View classification results and confidence scores |
| **🧠 GradCAM** | Visual explanation heatmap overlay |
| **⚠️ Anomaly** | OOD detection result and risk level |
| **📄 Report** | Download structured medical report |
| **📅 History** | Browse past prediction records |
| **📈 Benchmark** | View model performance comparison |

---

## 5. Running a Prediction

### Step-by-Step

**Step 1:** Click the **Upload Image** button in the Image Upload panel.

**Step 2:** Select a medical image file:
- Supported formats: **JPEG, PNG, BMP, TIFF, WEBP**
- Maximum file size: **20 MB**
- Minimum resolution: **32×32 pixels**

**Step 3:** Select the **Modality** from the dropdown:

| Modality | Description |
|---|---|
| Brain MRI | MRI scans of the brain |
| Brain CT | CT scans of the brain |
| Chest X-Ray | Chest radiography images |
| Retina | Fundus photography |
| Histopathology | Tissue biopsy microscopy |
| Skin Lesion | Dermoscopy images |
| Ultrasound | Sonography images |
| Auto-detect | Let the system classify the modality |

**Step 4:** (Optional) Enter a **Patient ID** for audit trail purposes.

**Step 5:** Click **Analyze** to run the prediction.

Processing typically completes within **100–200 milliseconds**.

---

## 6. Understanding Results

### 6.1 Prediction Result

The result panel displays:

| Field | Description |
|---|---|
| **Predicted Class** | The most likely disease condition |
| **Confidence Score** | Model confidence (0–100%) |
| **Processing Time** | Inference time in milliseconds |
| **Modality** | Detected or selected imaging modality |

### 6.2 Risk Level

Each prediction is assigned a risk tier based on the anomaly detection analysis:

| Risk Level | Color | Meaning |
|---|---|---|
| 🟢 LOW | Green | Normal finding, high confidence |
| 🟡 MEDIUM | Yellow | Borderline, clinical review recommended |
| 🔴 HIGH | Red | Significant anomaly, specialist required |
| 🚨 CRITICAL | Dark Red | Urgent clinical attention required |

### 6.3 GradCAM Heatmap

The heatmap overlay highlights regions of the image that most influenced the AI prediction. **Red/yellow areas** indicate the highest activation (most important regions for the prediction). Blue areas indicate low activation.

---

## 7. Medical Reports

Click **Download Report** to access the medical report in your preferred format:

| Format | Use Case |
|---|---|
| **PDF** | Formal clinical documentation |
| **Markdown** | EHR/EMR text attachment |
| **JSON** | System integration / data processing |

Reports include:
- Patient identifier and timestamp
- Imaging modality and predicted condition
- Confidence score and risk level
- GradCAM heatmap reference
- Anomaly score
- Recommended clinical action

---

## 8. Prediction History

The **History** panel displays a complete audit trail of all past predictions:

- Filter by Patient ID, Date Range, or Modality
- Click any record to view full details
- Export history as CSV for reporting

---

## 9. Benchmark Dashboard

The **Benchmark** tab displays comparative model performance metrics:

- Accuracy, Precision, Recall, F1 Score, AUC-ROC per modality
- Average inference time
- Side-by-side modality comparison chart

Click **Run New Benchmark** to re-evaluate all models against current test data.

---

## 10. REST API Usage

### Authentication

If API key authentication is enabled (default in production):

```bash
# Include API key in header
curl -H "X-API-Key: your_api_key" http://localhost:8000/predict
```

### Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your_api_key" \
  -F "file=@brain_mri.jpg" \
  -F "modality=brain_mri" \
  -F "patient_id=PAT-001"
```

### View API Documentation

Navigate to **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 11. Troubleshooting

### Platform will not start
- Ensure Python 3.9+ is installed: `python --version`
- Run setup: `pip install -r requirements.txt`
- Check port availability: ports 7860 and 8000 must be free

### Prediction returns an error
- Verify the image format is supported (JPEG, PNG, BMP, TIFF, WEBP)
- Ensure the image is not corrupted
- Check the image minimum size is at least 32×32 pixels

### Docker container will not start
- Ensure Docker Desktop is running
- Verify you have at least 4 GB free disk space
- Run `docker compose logs` to inspect errors

### Low confidence predictions
- Ensure you selected the correct modality
- Use high-quality images with clear anatomical features
- Avoid over-compressed or heavily artefacted images

---

## 12. Frequently Asked Questions

**Q: Is this platform FDA/CE approved?**
A: No. This is a research and educational platform. It is not approved for clinical use without appropriate regulatory validation.

**Q: What disease conditions does the platform detect?**
A: Over 40 conditions across 7 modalities. See the full list in the Technical Documentation.

**Q: Can I use my own trained models?**
A: Yes. Developers can integrate custom PyTorch models via the Modality Predictor plugin interface. See the Developer Manual for details.

**Q: How is patient data handled?**
A: No images are transmitted externally. All processing is local. Patient IDs are stored only in the local SQLite database.

**Q: How do I reset the prediction history?**
A: Delete the file `data/predictions.db`. The database will be recreated on next launch.
