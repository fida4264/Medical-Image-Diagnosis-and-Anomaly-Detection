# Module 11: Retina Disease Analysis Framework

## Overview

Module 11 implements the **Retina Disease Analysis Framework** for the Medical Image Diagnosis and Anomaly Detection System.
It accepts Retinal Fundus images automatically routed by Module 6 (`modality == 'retina'`),
dynamically discovers disease classes, performs multi-class classification, and provides **disease severity grading**.

---

## Dataset

| Field | Value |
|---|---|
| Modality Filter | `retina` (strict) |
| Total Records | 14,392 |
| Datasets | Ocular Disease Intelligent Recognition (ODIR-5K), Retina Fundus Images |
| Class Discovery | Fully dynamic from `master_metadata.csv` — zero hardcoded class names |
| Split Strategy | Full 70/15/15 stratified train/val/test |

### Supported Retinal Disease Classes (Dynamic Discovery)

Any class present in the retina metadata is automatically loaded. Examples include:
- `normal` / `no_dr`
- `diabetic_retinopathy` (mild, moderate, severe, proliferative)
- `glaucoma`
- `amd` / `age_related_macular_degeneration`
- `hypertensive_retinopathy`
- `retinal_vein_occlusion`
- Any additional labels present in the dataset

---

## Architecture

```
retina/
├── __init__.py
├── utils/
│   ├── config.py                — RetinaConfig (YAML dataclass)
│   ├── logger.py                — Re-exports Module 6 setup_logger
│   └── checkpoint_manager.py   — Save/load/resume manager
├── datasets/
│   ├── dataset_loader.py        — RetinaDatasetLoader (strict modality filter)
│   └── dataset_builder.py      — RetinaDataset + DataLoader factory
├── models/
│   ├── backbone_factory.py     — Re-exports Module 6 BackboneFactory
│   └── retina_model.py         — RetinaClassifier + build_retina_model
├── training/
│   ├── trainer.py              — RetinaTrainer (AMP + cosine LR + early stopping)
│   ├── callbacks.py            — Re-exports Module 6 EarlyStopping, ModelCheckpoint
│   └── metrics.py              — Re-exports Module 6 compute_classification_metrics
├── evaluation/
│   ├── evaluator.py            — RetinaEvaluator
│   ├── roc.py                  — Multi-class OvR ROC plot
│   ├── confusion_matrix.py     — Confusion matrix plot
│   ├── misclassified.py        — Misclassified sample grid
│   └── benchmark.py            — Latency benchmark
├── reports/
│   ├── clinical_report.py      — ClinicalReportGenerator (retina_report.json)
│   └── performance_report.py   — PerformanceReportGenerator (JSON + Markdown)
├── inference/
│   ├── predict_retina.py       — RetinaPredictor + RetinaPredictionResult + SeverityGrader
│   └── batch_predict.py        — BatchRetinaPredictor (CSV output)
└── export/
    ├── onnx_export.py          — RetinaONNXExporter (dynamo=False)
    └── torchscript_export.py   — TorchScriptExporter (torch.jit.trace)
```

---

## Disease Severity Grading

Module 11 introduces a dedicated **SeverityGrader** component that evaluates severity from:
1. **Stage keywords** in the predicted class name (`mild`, `moderate`, `severe`, `proliferative`)
2. **Grade stage tags** (`stage1`, `stage_1`, `stage2`, etc.)
3. **Prediction confidence** as a fallback for ungraded disease findings

### Severity Scale

| Grade | Stage | Description |
|---|---|---|
| Grade 0 | Normal | No apparent retinal disease |
| Grade 1 | Mild | Early stage — microaneurysms |
| Grade 2 | Moderate | Intermediate stage — haemorrhages |
| Grade 3 | Severe | Advanced stage — neovascularisation risk |
| Grade 4 | Proliferative | High-risk — active neovascularisation |
| Ungraded | Screening | Insufficient confidence — follow-up recommended |

---

## CLI Usage

### Full Training Pipeline

```bash
# Full train → evaluate → export pipeline
python run_retina.py

# Quick sample mode (50 images/class, 1 epoch)
python run_retina.py --sample --sample-size 50 --epochs 1

# Resume from last checkpoint
python run_retina.py --resume

# Custom backbone
python run_retina.py --backbone densenet121

# Override batch size and epochs
python run_retina.py --epochs 50 --batch-size 64
```

### Single-Image Prediction

```bash
# Predict retinal disease and severity stage
python predict_retina.py --image /path/to/fundus.jpg

# Save prediction JSON output
python predict_retina.py --image /path/to/fundus.jpg --output outputs/retina_pred.json

# Show top-5 predictions
python predict_retina.py --image /path/to/fundus.jpg --top-k 5
```

---

## Standardized Prediction JSON Schema

```json
{
  "module": "retina",
  "modality": "retina",
  "predicted_class": "diabetic_retinopathy",
  "confidence": 0.9412,
  "top3_predictions": [
    ["diabetic_retinopathy", 0.9412],
    ["normal", 0.031],
    ["glaucoma", 0.0278]
  ],
  "severity": "High Confidence Finding (Stage 2/3 - Clinical Correlation Required)",
  "probability_distribution": { ... },
  "model_version": "0.11.0",
  "inference_time_ms": 12.3,
  "timestamp": "2026-08-01T...",
  "status": "success"
}
```

---

## Configuration (configs/retina.yaml)

| Key | Default | Description |
|---|---|---|
| `model.backbone` | `efficientnet_b0` | Backbone architecture |
| `training.epochs` | `30` | Training epochs |
| `training.batch_size` | `32` | Batch size |
| `training.learning_rate` | `3e-4` | AdamW LR |
| `training.patience` | `7` | Early stopping patience |
| `training.use_amp` | `true` | Mixed precision (GPU only) |
| `severity.grading_enabled` | `true` | Enable severity grading |

---

## Output Files

| File | Description |
|---|---|
| `saved_models/retina/best_model.pt` | Best PyTorch checkpoint |
| `saved_models/retina/best_model.onnx` | ONNX export |
| `saved_models/retina/best_model.torchscript` | TorchScript export |
| `outputs/retina/roc_curves.png` | Multi-class ROC curves |
| `outputs/retina/confusion_matrix.png` | Confusion matrix |
| `outputs/retina/performance_report.json` | Full evaluation metrics |
| `outputs/retina/performance_report.md` | Human-readable report |
| `outputs/retina/reports/retina_report.json` | Clinical prediction report |
| `metadata/retina_splits.csv` | Train/val/test split |
| `metadata/retina_label_map.json` | Class name ↔ index map |

---

## Module 6 Components Reused

| Component | Source |
|---|---|
| `BackboneFactory` | `modality_classifier.models.backbone_factory` |
| `ModularClassifierHead` | `modality_classifier.models.classifier_head` |
| `EarlyStopping`, `ModelCheckpoint` | `modality_classifier.training.callbacks` |
| `compute_classification_metrics` | `modality_classifier.training.metrics` |
| `IMAGENET_MEAN`, `IMAGENET_STD` | `modality_classifier.datasets.data_loader` |
| `setup_logger` | `modality_classifier.utils.logging` |

---

## Future Integration

Module 11 outputs are fully compatible with:
- **Module 14 (Unified Prediction Engine):** `to_standard_dict()` schema with severity field.
- **Module 15 (Explainable AI):** `extract_features()` backbone method supports Grad-CAM.
- **Module 20 (Gradio UI):** `RetinaPredictor.predict_image()` accepts PIL images directly.

---

*Module 11: Retina Disease Analysis Framework — v0.11.0*
