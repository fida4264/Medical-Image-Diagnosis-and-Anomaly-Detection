# Module 13: Ultrasound Disease Analysis Framework

## Overview

Module 13 implements the **Ultrasound Disease Analysis Framework** for the Medical Image Diagnosis and Anomaly Detection System.
It accepts Ultrasound images automatically routed by Module 6 (`modality == 'ultrasound'`),
dynamically discovers disease classes, performs multi-class classification, and provides **clinical prediction reliability scoring**.

Module 13 completes the collection of 6 modality-specific AI engines in Phase 2.

---

## Dataset

| Field | Value |
|---|---|
| Modality Filter | `ultrasound` (strict) |
| Total Records | 780 |
| Datasets | Breast Ultrasound Images Dataset (BUSI) |
| Class Discovery | Fully dynamic from `master_metadata.csv` — zero hardcoded class names |
| Split Strategy | Full 70/15/15 stratified train/val/test |

### Discovered Ultrasound Disease Classes

- `benign` (437 samples) — Benign breast mass
- `malignant` (210 samples) — Malignant breast neoplasm (High Risk)
- `normal` (133 samples) — Normal breast ultrasound study

---

## Architecture

```
ultrasound/
├── __init__.py
├── utils/
│   ├── config.py                — UltrasoundConfig (YAML dataclass)
│   ├── logger.py                — Re-exports Module 6 setup_logger
│   └── checkpoint_manager.py   — Save/load/resume manager
├── datasets/
│   ├── dataset_loader.py        — UltrasoundDatasetLoader (strict modality filter)
│   └── dataset_builder.py      — UltrasoundDataset + DataLoader factory
├── models/
│   ├── backbone_factory.py     — Re-exports Module 6 BackboneFactory
│   └── ultrasound_model.py     — UltrasoundClassifier + build_ultrasound_model
├── training/
│   ├── trainer.py              — UltrasoundTrainer (AMP + cosine LR + early stopping)
│   ├── callbacks.py            — Re-exports Module 6 EarlyStopping, ModelCheckpoint
│   └── metrics.py              — Re-exports Module 6 compute_classification_metrics
├── evaluation/
│   ├── evaluator.py            — UltrasoundEvaluator
│   ├── roc.py                  — Multi-class OvR ROC plot
│   ├── confusion_matrix.py     — Confusion matrix plot
│   ├── misclassified.py        — Misclassified sample grid
│   └── benchmark.py            — Latency benchmark
├── reports/
│   ├── clinical_report.py      — ClinicalReportGenerator (ultrasound_report.json)
│   └── performance_report.py   — PerformanceReportGenerator (JSON + Markdown)
├── inference/
│   ├── predict_ultrasound.py   — UltrasoundPredictor + PredictionReliabilityScorer
│   └── batch_predict.py        — BatchUltrasoundPredictor (CSV output)
└── export/
    ├── onnx_export.py          — UltrasoundONNXExporter (dynamo=False)
    └── torchscript_export.py   — TorchScriptExporter (torch.jit.trace)
```

---

## Clinical Prediction Reliability Scoring

Module 13 includes a dedicated **PredictionReliabilityScorer** that evaluates prediction confidence and flags low-confidence predictions for mandatory human expert review:

| Confidence | Reliability | Requires Manual Review |
|---|---|---|
| `≥ 85%` | `high` | `false` |
| `65% – 84%` | `moderate` | `false` |
| `< 65%` | `low` | `true` |

---

## Standardized Prediction JSON Schema

```json
{
  "module": "ultrasound",
  "modality": "ultrasound",
  "predicted_class": "malignant",
  "confidence": 0.9412,
  "prediction_reliability": "high",
  "requires_manual_review": false,
  "top3_predictions": [
    ["malignant", 0.9412],
    ["benign", 0.041],
    ["normal", 0.0178]
  ],
  "probability_distribution": { ... },
  "model_version": "0.13.0",
  "inference_time_ms": 11.2,
  "timestamp": "2026-08-01T...",
  "status": "success"
}
```

---

## CLI Usage

### Full Training Pipeline

```bash
# Full train → evaluate → export pipeline
python run_ultrasound.py

# Quick sample mode (50 images/class, 1 epoch)
python run_ultrasound.py --sample --sample-size 50 --epochs 1

# Resume from last checkpoint
python run_ultrasound.py --resume

# Custom backbone
python run_ultrasound.py --backbone densenet121

# Override batch size and epochs
python run_ultrasound.py --epochs 50 --batch-size 64
```

### Single-Image Prediction

```bash
# Predict ultrasound finding and reliability
python predict_ultrasound.py --image /path/to/us.png

# Save prediction JSON output
python predict_ultrasound.py --image /path/to/us.png --output outputs/us_pred.json

# Show top-3 predictions
python predict_ultrasound.py --image /path/to/us.png --top-k 3
```

---

## Configuration (configs/ultrasound.yaml)

| Key | Default | Description |
|---|---|---|
| `model.backbone` | `efficientnet_b0` | Backbone architecture |
| `training.epochs` | `30` | Training epochs |
| `training.batch_size` | `32` | Batch size |
| `training.learning_rate` | `3e-4` | AdamW LR |
| `training.patience` | `7` | Early stopping patience |
| `training.use_amp` | `true` | Mixed precision (GPU only) |
| `reliability.high_threshold` | `0.85` | High confidence threshold |
| `reliability.moderate_threshold` | `0.65` | Moderate confidence threshold |

---

## Output Files

| File | Description |
|---|---|
| `saved_models/ultrasound/best_model.pt` | Best PyTorch checkpoint |
| `saved_models/ultrasound/best_model.onnx` | ONNX export |
| `saved_models/ultrasound/best_model.torchscript` | TorchScript export |
| `outputs/ultrasound/roc_curves.png` | Multi-class ROC curves |
| `outputs/ultrasound/confusion_matrix.png` | Confusion matrix |
| `outputs/ultrasound/performance_report.json` | Full evaluation metrics |
| `outputs/ultrasound/performance_report.md` | Human-readable report |
| `outputs/ultrasound/reports/ultrasound_report.json` | Clinical prediction report |
| `metadata/ultrasound_splits.csv` | Train/val/test split |
| `metadata/ultrasound_label_map.json` | Class name ↔ index map |

---

## Complete AI Pipeline Summary (Phase 2 Completed)

| Module | Modality | Supported Classes | Key Feature |
|---|---|---|---|
| Module 6 | Modality Classifier | 6 modalities | Auto-routing engine |
| Module 7 | Brain MRI | 4 classes | Reference Template |
| Module 8 | Chest X-ray | 3 classes | Multi-disease classification |
| Module 9 | Brain CT | Image representation | Feature extraction engine |
| Module 10 | Histopathology | 5 classes | Cancer subtyping |
| Module 11 | Retina | Dynamic classes | Disease Severity Grader |
| Module 12 | Skin Lesion | 7 classes | HAM10000 Label Recovery & Reliability Scorer |
| Module 13 | Ultrasound | 3 classes | BUSI Breast Lesion Diagnosis & Reliability Scorer |

---

*Module 13: Ultrasound Disease Analysis Framework — v0.13.0*
