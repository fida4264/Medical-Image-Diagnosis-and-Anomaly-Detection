# Module 8: Chest X-ray Disease Analysis Framework

## Overview

Module 8 implements the **Chest X-ray Disease Analysis Framework**, extending the core medical AI architecture established in Module 7. It accepts processed Chest X-ray images, automatically discovers disease classes present in the metadata (zero hardcoded class names), trains a deep transfer learning model, performs full statistical evaluation, exports production models (ONNX & TorchScript), generates standardized prediction JSON schemas (for Module 14 integration), and produces comprehensive clinical reports.

---

## 1. Architecture & Workflow

```
[Processed Images / Metadata]
         │ (modality == "chest_xray")
         ▼
[ChestXrayDatasetLoader]  ──► Dynamic Disease Class Discovery (e.g. covid, normal, pneumonia, etc.)
         │
         ▼
[ChestXrayDataset / DataLoader]
         │
         ▼
[ChestXrayClassifier]
   ├── Backbone (Default: EfficientNet-B0 / ResNet / DenseNet / MobileNet / ConvNeXt)
   ├── AdaptiveAvgPool2d
   ├── BatchNorm
   └── ModularClassifierHead ──► Logits (dim = num_discovered_classes)
         │
         ▼
[ChestXrayTrainer] (AMP Mixed Precision, EarlyStopping, ReduceLROnPlateau, TensorBoard)
         │
         ├──► [Best Model Checkpoints (.pt)]
         ├──► [ONNX Export (.onnx)]
         └──► [TorchScript Export (.torchscript)]
         │
         ▼
[ChestXrayEvaluator] ──► Confusion Matrix, ROC Curves, Misclassified Grids, Latency Benchmark
         │
         ▼
[ClinicalReportGenerator / ChestXrayPredictor]
         ├──► chest_xray_report.json (Clinical Findings & Differential Diagnosis)
         └──► Standardized Prediction JSON (Unified schema for Module 14 Engine)
```

---

## 2. Dynamic Disease Class Discovery

Module 8 **never hardcodes disease names**. During initialization, `ChestXrayDatasetLoader` queries `master_metadata.csv` for records matching `modality == 'chest_xray'`, extracts unique values from `normalized_class`, sorts them, and creates an automatic label mapping:

```json
{
  "label_to_class": {
    "0": "covid",
    "1": "lung_opacity",
    "2": "normal",
    "3": "unknown",
    "4": "viral_pneumonia"
  },
  "class_to_label": {
    "covid": 0,
    "lung_opacity": 1,
    "normal": 2,
    "unknown": 3,
    "viral_pneumonia": 4
  },
  "num_classes": 5,
  "modality": "chest_xray"
}
```

The output layer of `ChestXrayClassifier` dynamically sets its neuron count to match `num_classes`.

---

## 3. Standardized Prediction JSON Schema

Module 8 establishes the unified output format required for compatibility with **Module 14 (Unified Prediction Engine)**:

```json
{
  "module": "chest_xray",
  "modality": "chest_xray",
  "predicted_class": "covid",
  "confidence": 0.954212,
  "top3_predictions": [
    {
      "rank": 1,
      "disease": "covid",
      "confidence": 0.954212
    },
    {
      "rank": 2,
      "disease": "viral_pneumonia",
      "confidence": 0.032104
    },
    {
      "rank": 3,
      "disease": "normal",
      "confidence": 0.013684
    }
  ],
  "model_version": "0.8.0",
  "inference_time_ms": 14.25,
  "timestamp": "2026-08-01T13:49:00Z",
  "image_path": "processed_images/chest_xray/covid/CX_001.png",
  "status": "success",
  "error_message": ""
}
```

---

## 4. Execution & CLI Entrypoints

### Pipeline Training, Evaluation, and Export
```bash
# Run full pipeline with default configuration (configs/chest_xray.yaml)
python run_chest_xray.py

# Quick verification run on sample dataset subset
python run_chest_xray.py --sample --sample-size 50 --epochs 1

# Resume training from last checkpoint
python run_chest_xray.py --resume

# Specify different backbone architecture
python run_chest_xray.py --backbone densenet121
```

### Single Image Inference
```bash
# Run disease prediction on a single Chest X-ray image
python predict_chest_xray.py --image path/to/chest_xray_scan.png

# Predict and generate clinical JSON report
python predict_chest_xray.py --image path/to/chest_xray_scan.png --report
```

---

## 5. Artifacts & Outputs

All generated assets are saved to `saved_models/chest_xray/`:

```
saved_models/chest_xray/
├── best_model.pt                      # Best model checkpoint
├── last_model.pt                      # Resume checkpoint
├── best_model.onnx                    # Production ONNX model
├── best_model.onnx.json               # ONNX metadata
├── best_model.torchscript             # TorchScript model
├── classification_report.json        # Detailed evaluation metrics
├── confusion_matrix.png               # Dual raw/normalized confusion matrix plot
├── roc_curves/
│   └── chest_xray_roc_curves.png     # Multi-class ROC curves
├── tensorboard_logs/                  # TensorBoard execution events
├── evaluation/
│   ├── performance_report.md          # Markdown performance summary
│   └── misclassified/
│       └── grid.png                   # Misclassified sample grid
└── reports/
    └── chest_xray_report.json        # Clinical findings report
```

---

## 6. Testing & Quality Assurance

Run the comprehensive unit test suite:
```bash
# Test Module 8 specifically
pytest tests/test_chest_xray.py -v

# Run full system test suite (Modules 1–8)
pytest tests/ -v
```
