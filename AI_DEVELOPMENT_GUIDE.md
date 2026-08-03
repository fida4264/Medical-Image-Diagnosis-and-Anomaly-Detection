# AI Development Guide — Implementing Disease-Specific Modules (Modules 8–13)

This guide provides instructions for implementing disease-specific modules (Modules 8–13) using **Module 7 (`brain_mri/`)** as the reference template.

---

## 1. Golden Rules for Disease Modules

1. **Re-use Module 6 Components:** Import `BackboneFactory`, `ModularClassifierHead`, `EarlyStopping`, `ModelCheckpoint`, `compute_classification_metrics`, and `IMAGENET_MEAN/STD` directly.
2. **Re-use Module 7 Structure:** Follow the exact 9-package directory structure of `brain_mri/`.
3. **Dynamic Disease Class Discovery:** Never hardcode disease class names. Always read them dynamically from metadata.
4. **Strict Modality Filtering:** Filter strictly for `modality == <target_modality>`. Reject all other modalities.
5. **Standardized Clinical Reports:** Generate structured JSON clinical diagnostic reports containing predicted_disease, confidence, probability_distribution, top_3_predictions, model_version, inference_time_ms, prediction_timestamp.
6. **Standardized Prediction Schema:** Produce JSON with `module`, `modality`, `predicted_class`, `confidence`, `top3_predictions`, `model_version`, `inference_time_ms`, `timestamp`, `status`. Add module-specific fields (e.g., `severity` for Module 11, `prediction_reliability` & `requires_manual_review` for Modules 12 & 13).
7. **Use `dynamo=False` for ONNX export:** Always pass `dynamo=False` to `torch.onnx.export()` to avoid `onnxscript` dependency.
8. **Pass probability arrays to metrics:** `compute_classification_metrics(y_true, y_pred_probs_2d_array)` — second argument must be the full 2D probability matrix, not predicted class indices.

---

## 2. Directory Structure Template

```
<modality_name>/
├── __init__.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── checkpoint_manager.py
├── datasets/
│   ├── dataset_loader.py
│   └── dataset_builder.py
├── models/
│   ├── <modality>_model.py
│   └── backbone_factory.py (thin re-export)
├── training/
│   ├── trainer.py
│   ├── callbacks.py (thin re-export)
│   └── metrics.py
├── evaluation/
│   ├── evaluator.py
│   ├── roc.py
│   ├── confusion_matrix.py
│   ├── misclassified.py
│   └── benchmark.py
├── reports/
│   ├── clinical_report.py
│   └── performance_report.py
├── inference/
│   ├── predict_<modality>.py
│   └── batch_predict.py
└── export/
    ├── onnx_export.py
    └── torchscript_export.py
```

---

## 3. Completed Module Reference Implementations (Phase 2 Complete)

| Module | Package | Modality Filter | Special Feature |
|---|---|---|---|
| Module 7 | `brain_mri/` | `brain_mri` | Reference Template |
| Module 8 | `chest_xray/` | `chest_xray` | — |
| Module 9 | `brain_ct/` | `brain_ct` | Feature extraction (no disease labels) |
| Module 10 | `histopathology/` | `histopathology` | LC25000 dataset, 5 cancer subtypes |
| Module 11 | `retina/` | `retina` | Disease Severity Grader (Grade 0–4) |
| Module 12 | `skin/` | `skin` | HAM10000 Label Recovery Join & Reliability Scorer |
| Module 13 | `ultrasound/` | `ultrasound` | BUSI Breast Lesion Diagnosis & Reliability Scorer |

---

## 4. Critical API Notes

### compute_classification_metrics
```python
# CORRECT: pass full probability matrix (2D numpy array)
metrics = compute_classification_metrics(y_true_labels, y_pred_probs)  # y_pred_probs.shape = (N, num_classes)
```

### torch.onnx.export
```python
# ALWAYS include dynamo=False to avoid onnxscript dependency
torch.onnx.export(model, dummy_input, path, ..., dynamo=False)
```

### IMAGENET_MEAN / IMAGENET_STD
```python
from modality_classifier.datasets.data_loader import IMAGENET_MEAN, IMAGENET_STD
```
