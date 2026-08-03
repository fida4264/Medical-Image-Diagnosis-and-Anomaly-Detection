# Module 10: Histopathology Cancer Subtyping Framework

## Overview

Module 10 implements **Histopathology Cancer Subtyping** for the Medical Image Diagnosis and Anomaly Detection Framework.
It uses the **LC25000 (Lung and Colon Cancer)** dataset and follows the exact architecture established in Module 7 (Brain MRI).

---

## Dataset

| Field | Value |
|---|---|
| Dataset | Lung and Colon Cancer Histopathological Images (LC25000) |
| Total Images | 25,000 |
| Cancer Subtypes | 5 (dynamically discovered) |
| Class Balance | Perfectly balanced (5,000/class) |
| Modality Filter | `histopathology` (strict) |

### Discovered Cancer Subtypes

| Index | Class Name | Description |
|---|---|---|
| 0 | `colon_adenocarcinoma` | Malignant glandular colon tumor |
| 1 | `colon_normal` | Normal colon tissue |
| 2 | `lung_adenocarcinoma` | Most common lung cancer subtype |
| 3 | `lung_normal` | Normal lung tissue |
| 4 | `lung_squamous_cell` | Central-location lung cancer |

---

## Architecture

```
histopathology/
├── __init__.py
├── utils/
│   ├── config.py              — HistopathologyConfig (YAML dataclass)
│   ├── logger.py              — Re-exports Module 6 setup_logger
│   └── checkpoint_manager.py — Save/load/resume manager
├── datasets/
│   ├── dataset_loader.py      — HistopathologyDatasetLoader (strict modality filter)
│   └── dataset_builder.py     — HistopathologyDataset + DataLoader factory
├── models/
│   ├── backbone_factory.py    — Re-exports Module 6 BackboneFactory
│   └── histopathology_model.py — HistopathologyClassifier
├── training/
│   ├── trainer.py             — HistopathologyTrainer (AMP + cosine LR + early stopping)
│   ├── callbacks.py           — Re-exports Module 6 EarlyStopping, ModelCheckpoint
│   └── metrics.py             — Re-exports Module 6 compute_classification_metrics
├── evaluation/
│   ├── evaluator.py           — HistopathologyEvaluator
│   ├── roc.py                 — Multi-class OvR ROC plot
│   ├── confusion_matrix.py    — Confusion matrix plot
│   ├── misclassified.py       — Misclassified sample grid
│   └── benchmark.py           — Latency benchmark
├── reports/
│   ├── clinical_report.py     — ClinicalReportGenerator (JSON)
│   └── performance_report.py  — PerformanceReportGenerator (JSON + Markdown)
├── inference/
│   ├── predict_histopathology.py — HistopathologyPredictor + HistopathologyPredictionResult
│   └── batch_predict.py       — BatchHistopathologyPredictor (CSV output)
└── export/
    ├── onnx_export.py         — HistopathologyONNXExporter
    └── torchscript_export.py  — TorchScriptExporter
```

---

## CLI Usage

### Full Training Pipeline

```bash
# Full train → evaluate → export pipeline
python run_histopathology.py

# Quick sample mode (50 images/class, 1 epoch)
python run_histopathology.py --sample --sample-size 50 --epochs 1

# Resume from last checkpoint
python run_histopathology.py --resume

# Custom backbone
python run_histopathology.py --backbone densenet121

# Override batch size and epochs
python run_histopathology.py --epochs 50 --batch-size 64
```

### Single-Image Prediction

```bash
# Predict cancer subtype from an image
python predict_histopathology.py --image /path/to/slide.png

# Save prediction JSON output
python predict_histopathology.py --image /path/to/slide.png --output outputs/prediction.json

# Show top-5 predictions
python predict_histopathology.py --image /path/to/slide.png --top-k 5
```

---

## Configuration

Configuration is loaded from `configs/histopathology.yaml`. All fields have sensible defaults.

| Key | Default | Description |
|---|---|---|
| `model.backbone` | `efficientnet_b0` | Backbone architecture |
| `training.epochs` | `30` | Training epochs |
| `training.batch_size` | `32` | Batch size |
| `training.learning_rate` | `3e-4` | AdamW LR |
| `training.patience` | `7` | Early stopping patience |
| `training.use_amp` | `true` | Mixed precision (GPU only) |
| `evaluation.generate_roc` | `true` | ROC curve generation |
| `evaluation.generate_confusion_matrix` | `true` | Confusion matrix |

---

## Prediction JSON Schema

```json
{
  "module": "histopathology",
  "modality": "histopathology",
  "predicted_class": "lung_adenocarcinoma",
  "confidence": 0.9412,
  "probability_distribution": {
    "colon_adenocarcinoma": 0.01,
    "colon_normal": 0.005,
    "lung_adenocarcinoma": 0.9412,
    "lung_normal": 0.009,
    "lung_squamous_cell": 0.034
  },
  "top_3_predictions": [
    ["lung_adenocarcinoma", 0.9412],
    ["lung_squamous_cell", 0.034],
    ["colon_adenocarcinoma", 0.01]
  ],
  "model_version": "0.10.0",
  "inference_time_ms": 11.2,
  "prediction_timestamp": "2026-08-01T..."
}
```

---

## Output Files

| File | Description |
|---|---|
| `saved_models/histopathology/best_model.pt` | Best PyTorch checkpoint |
| `saved_models/histopathology/best_model.onnx` | ONNX export |
| `saved_models/histopathology/best_model.torchscript` | TorchScript export |
| `outputs/histopathology/roc_curves.png` | Multi-class ROC curves |
| `outputs/histopathology/confusion_matrix.png` | Confusion matrix |
| `outputs/histopathology/performance_report.json` | Full evaluation metrics |
| `outputs/histopathology/performance_report.md` | Human-readable report |
| `outputs/histopathology/reports/histopathology_report.json` | Clinical prediction report |
| `metadata/histopathology_splits.csv` | Train/val/test split |
| `metadata/histopathology_label_map.json` | Class name ↔ index map |

---

## Module 6 Components Reused

| Component | Source |
|---|---|
| `BackboneFactory` | `modality_classifier.models.backbone_factory` |
| `ModularClassifierHead` | `modality_classifier.models.classifier_head` |
| `EarlyStopping`, `ModelCheckpoint` | `modality_classifier.training.callbacks` |
| `compute_classification_metrics` | `modality_classifier.training.metrics` |
| `IMAGENET_MEAN`, `IMAGENET_STD` | `modality_classifier.models.backbone_factory` |
| `setup_logger` | `modality_classifier.utils.logging` |

---

*Module 10: Histopathology Cancer Subtyping Framework — v0.10.0*
