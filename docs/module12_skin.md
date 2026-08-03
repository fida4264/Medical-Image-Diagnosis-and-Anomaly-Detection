# Module 12: Skin Lesion Cancer Classification Framework

## Overview

Module 12 implements the **Skin Lesion Cancer Classification Framework** for the Medical Image Diagnosis and Anomaly Detection System.
It accepts Skin Lesion images automatically routed by Module 6 (`modality == 'skin'`), recovers HAM10000 diagnosis codes, dynamically discovers skin lesion classes, performs multi-class classification, and provides **clinical prediction reliability scoring**.

---

## Dataset

| Field | Value |
|---|---|
| Modality Filter | `skin` (strict) |
| Total Records | 10,015 |
| Primary Dataset | HAM10000 (Human Against Machine with 10,000 training images) |
| Label Recovery | Joined with `datasets/skin/HAM10000_metadata.csv` to recover exact `dx` codes |
| Class Discovery | Fully dynamic from metadata — zero hardcoded class names |
| Split Strategy | Full 70/15/15 stratified train/val/test |

### Discovered HAM10000 Skin Lesion Classes

| Code | Class Name | Category |
|---|---|---|
| `nv` | `melanocytic_nevus` | Benign |
| `mel` | `melanoma` | **Malignant** (High Risk) |
| `bkl` | `benign_keratosis` | Benign |
| `bcc` | `basal_cell_carcinoma` | **Malignant** |
| `akiec` | `actinic_keratosis` | Pre-malignant / In-situ |
| `vasc` | `vascular_lesion` | Benign |
| `df` | `dermatofibroma` | Benign |

---

## Architecture

```
skin/
├── __init__.py
├── utils/
│   ├── config.py                — SkinLesionConfig (YAML dataclass)
│   ├── logger.py                — Re-exports Module 6 setup_logger
│   └── checkpoint_manager.py   — Save/load/resume manager
├── datasets/
│   ├── dataset_loader.py        — SkinLesionDatasetLoader (HAM10000 label recovery join)
│   └── dataset_builder.py      — SkinLesionDataset + DataLoader factory
├── models/
│   ├── backbone_factory.py     — Re-exports Module 6 BackboneFactory
│   └── skin_model.py           — SkinLesionClassifier + build_skin_model
├── training/
│   ├── trainer.py              — SkinLesionTrainer (AMP + cosine LR + early stopping)
│   ├── callbacks.py            — Re-exports Module 6 EarlyStopping, ModelCheckpoint
│   └── metrics.py              — Re-exports Module 6 compute_classification_metrics
├── evaluation/
│   ├── evaluator.py            — SkinLesionEvaluator
│   ├── roc.py                  — Multi-class OvR ROC plot
│   ├── confusion_matrix.py     — Confusion matrix plot
│   ├── misclassified.py        — Misclassified sample grid
│   └── benchmark.py            — Latency benchmark
├── reports/
│   ├── clinical_report.py      — ClinicalReportGenerator (skin_report.json)
│   └── performance_report.py   — PerformanceReportGenerator (JSON + Markdown)
├── inference/
│   ├── predict_skin.py         — SkinLesionPredictor + PredictionReliabilityScorer
│   └── batch_predict.py        — BatchSkinPredictor (CSV output)
└── export/
    ├── onnx_export.py          — SkinONNXExporter (dynamo=False)
    └── torchscript_export.py   — TorchScriptExporter (torch.jit.trace)
```

---

## Clinical Prediction Reliability Scoring

Module 12 includes a dedicated **PredictionReliabilityScorer** that evaluates prediction confidence and flags low-confidence predictions for mandatory human expert review:

| Confidence | Reliability | Requires Manual Review |
|---|---|---|
| `≥ 85%` | `high` | `false` |
| `65% – 84%` | `moderate` | `false` |
| `< 65%` | `low` | `true` |

---

## Standardized Prediction JSON Schema

```json
{
  "module": "skin",
  "modality": "skin",
  "predicted_class": "melanoma",
  "confidence": 0.9312,
  "top3_predictions": [
    ["melanoma", 0.9312],
    ["melanocytic_nevus", 0.041],
    ["basal_cell_carcinoma", 0.021]
  ],
  "prediction_reliability": "high",
  "requires_manual_review": false,
  "probability_distribution": { ... },
  "model_version": "0.12.0",
  "inference_time_ms": 11.8,
  "timestamp": "2026-08-01T...",
  "status": "success"
}
```

---

## CLI Usage

### Full Training Pipeline

```bash
# Full train → evaluate → export pipeline
python run_skin.py

# Quick sample mode (50 images/class, 1 epoch)
python run_skin.py --sample --sample-size 50 --epochs 1

# Resume from last checkpoint
python run_skin.py --resume

# Custom backbone
python run_skin.py --backbone densenet121

# Override batch size and epochs
python run_skin.py --epochs 50 --batch-size 64
```

### Single-Image Prediction

```bash
# Predict skin lesion disease and reliability
python predict_skin.py --image /path/to/lesion.jpg

# Save prediction JSON output
python predict_skin.py --image /path/to/lesion.jpg --output outputs/skin_pred.json

# Show top-5 predictions
python predict_skin.py --image /path/to/lesion.jpg --top-k 5
```

---

## Configuration (configs/skin_lesion.yaml)

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
| `saved_models/skin/best_model.pt` | Best PyTorch checkpoint |
| `saved_models/skin/best_model.onnx` | ONNX export |
| `saved_models/skin/best_model.torchscript` | TorchScript export |
| `outputs/skin/roc_curves.png` | Multi-class ROC curves |
| `outputs/skin/confusion_matrix.png` | Confusion matrix |
| `outputs/skin/performance_report.json` | Full evaluation metrics |
| `outputs/skin/performance_report.md` | Human-readable report |
| `outputs/skin/reports/skin_report.json` | Clinical prediction report |
| `metadata/skin_splits.csv` | Train/val/test split |
| `metadata/skin_label_map.json` | Class name ↔ index map |

---

## Future Integration

Module 12 outputs are fully compatible with:
- **Module 14 (Unified Prediction Engine):** `to_standard_dict()` schema with reliability fields.
- **Module 15 (Explainable AI):** `extract_features()` backbone method supports Grad-CAM.
- **Module 20 (Gradio UI):** `SkinLesionPredictor.predict_image()` accepts PIL images directly.

---

*Module 12: Skin Lesion Cancer Classification Framework — v0.12.0*
