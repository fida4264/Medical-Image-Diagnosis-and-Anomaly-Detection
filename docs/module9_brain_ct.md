# Module 9: Brain CT Image Analysis Framework

## Overview

Module 9 implements the **Brain CT Image Analysis Framework**, extending the core medical AI architecture established in Phase 2. Designed specifically for Brain CT scans, this framework automatically loads processed Brain CT images (routed via Module 6), extracts deep structural visual representations, supports transfer learning feature extraction, performs statistical evaluation, exports production ONNX and TorchScript models, and generates standardized prediction JSON outputs (`brain_ct_prediction.json`) and clinical reports (`brain_ct_report.json`).

> **Framework Objective:** Brain CT images in the current dataset represent domain anatomical scans (`head_ct`). Rather than inventing synthetic disease labels, Module 9 is optimized to learn robust Brain CT representations, provide spatial feature maps for future Grad-CAM visual explanations (Module 15), support unsupervised anomaly detection (Module 16), and integrate seamlessly with the Unified Prediction Engine (Module 14).

---

## 1. Architecture & Workflow

```
[Processed Images / Metadata]
         │ (modality == "brain_ct")
         ▼
[BrainCTDatasetLoader]   ──► Dynamic Representation & Category Discovery (e.g. head_ct)
         │
         ▼
[BrainCTDataset / DataLoader]
         │
         ▼
[BrainCTClassifier]
   ├── Transfer Learning Backbone (Default: EfficientNet-B0 / ResNet / DenseNet / MobileNet / ConvNeXt)
   ├── Spatial Feature Map Extractor (Grad-CAM Ready for Module 15)
   ├── AdaptiveAvgPool2d + BatchNorm + Dropout
   └── Modular Head ──► Output Logits (dim = num_discovered_categories)
         │
         ▼
[BrainCTTrainer] (AMP Mixed Precision, EarlyStopping, ReduceLROnPlateau, TensorBoard)
         │
         ├──► [Best Model Checkpoints (.pt)]
         ├──► [ONNX Export (.onnx)]
         └──► [TorchScript Export (.torchscript)]
         │
         ▼
[BrainCTEvaluator] ──► Confusion Matrix, ROC Curves, Misclassified Grids, Latency Benchmark
         │
         ▼
[ClinicalReportGenerator / BrainCTPredictor]
         ├──► brain_ct_report.json (Clinical Findings & Representation Analysis)
         └──► Standardized Prediction JSON (Unified schema for Module 14 Engine)
```

---

## 2. Dynamic Category & Representation Discovery

Module 9 **never hardcodes class names**. During initialization, `BrainCTDatasetLoader` queries `master_metadata.csv` for records matching `modality == 'brain_ct'`, extracts unique values from `normalized_class`, sorts them, and creates an automatic label mapping:

```json
{
  "label_to_class": {
    "0": "head_ct"
  },
  "class_to_label": {
    "head_ct": 0
  },
  "num_classes": 1,
  "modality": "brain_ct"
}
```

The output head of `BrainCTClassifier` dynamically adapts its dimensions to match the discovered categories.

---

## 3. Standardized Prediction JSON Schema (`brain_ct_prediction.json`)

Module 9 maintains the unified output format required for compatibility with **Module 14 (Unified Prediction Engine)**:

```json
{
  "module": "brain_ct",
  "modality": "brain_ct",
  "predicted_class": "head_ct",
  "confidence": 0.9854,
  "top3_predictions": [
    {
      "rank": 1,
      "disease": "head_ct",
      "confidence": 0.9854
    }
  ],
  "model_version": "0.9.0",
  "inference_time_ms": 11.85,
  "timestamp": "2026-08-01T13:58:00Z",
  "image_path": "processed_images/brain_ct/head_ct/CT_001.png",
  "status": "success",
  "error_message": ""
}
```

---

## 4. Execution & CLI Entrypoints

### Pipeline Training, Evaluation, and Export
```bash
# Run full pipeline with default configuration (configs/brain_ct.yaml)
python run_brain_ct.py

# Quick verification run on sample dataset subset
python run_brain_ct.py --sample --sample-size 50 --epochs 1

# Resume training from last checkpoint
python run_brain_ct.py --resume

# Specify different backbone architecture
python run_brain_ct.py --backbone resnet18
```

### Single Image Inference
```bash
# Run representation & category prediction on a single Brain CT scan
python predict_brain_ct.py --image path/to/brain_ct_scan.png

# Predict and generate clinical JSON report
python predict_brain_ct.py --image path/to/brain_ct_scan.png --report
```

---

## 5. Output Directory Structure

All generated assets are saved to `saved_models/brain_ct/`:

```
saved_models/brain_ct/
├── best_model.pt                      # Best model checkpoint
├── last_model.pt                      # Resume checkpoint
├── best_model.onnx                    # Production ONNX model
├── best_model.onnx.json               # ONNX metadata
├── best_model.torchscript             # TorchScript model
├── classification_report.json        # Detailed evaluation metrics
├── confusion_matrix.png               # Dual raw/normalized confusion matrix plot
├── roc_curves/
│   └── brain_ct_roc_curves.png        # Multi-class ROC curves
├── tensorboard_logs/                  # TensorBoard execution events
├── evaluation/
│   ├── performance_report.md          # Markdown performance summary
│   └── misclassified/
│       └── grid.png                   # Misclassified sample grid
└── reports/
    └── brain_ct_report.json           # Clinical findings report
```

---

## 6. Testing & Quality Assurance

Run the comprehensive unit test suite:
```bash
# Test Module 9 specifically
pytest tests/test_brain_ct.py -v

# Run full system test suite (Modules 1–9)
pytest tests/ -v
```
