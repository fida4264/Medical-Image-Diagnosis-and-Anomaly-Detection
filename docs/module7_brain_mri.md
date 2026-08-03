# Module 7 Design Specification — Brain MRI Disease Analysis Framework

---

## 1. Executive Summary

Module 7 is the **Brain MRI Disease Analysis Framework**. It receives images routed from Module 6 (where `modality == brain_mri`) and identifies the specific disease present.

It serves as the **reference implementation template** for all future disease-specific modules (Modules 8–13).

### Target Disease Classes (Dynamically Discovered)
1. `glioma` — Glioma brain tumor
2. `meningioma` — Meningioma brain tumor
3. `pituitary` — Pituitary tumor
4. `notumor` — Normal / No tumor detected

---

## 2. Architectural Design

```
Preprocessed Brain MRI (224x224x3)
              │
              ▼
┌─────────────────────────────────────────────┐
│ Transfer Learning Backbone                  │
│ (EfficientNet-B0 / ResNet18 / DenseNet121)  │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Global Adaptive Average Pooling (GAP)       │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ BatchNorm1D + Dropout (0.4) + Dense Softmax │
└─────────────────────────────────────────────┘
              │
              ▼
       Disease Softmax Logits
              │
              ▼
   Clinical Report Generator (brain_mri_report.json)
```

---

## 3. Directory Structure

```
brain_mri/
├── __init__.py
├── utils/
│   ├── config.py                 ← BrainMRIConfig Dataclass
│   ├── logger.py                 ← Configured Logger (logs/brain_mri.log)
│   └── checkpoint_manager.py     ← Best & Last Checkpoint Recovery
├── datasets/
│   ├── dataset_loader.py         ← Strict brain_mri filter & Splitter
│   └── dataset_builder.py        ← BrainMRIDataset & DataLoaders
├── models/
│   ├── brain_mri_model.py        ← BrainMRIClassifier PyTorch Module
│   └── backbone_factory.py       ← Thin Re-export of Module 6 BackboneFactory
├── training/
│   ├── trainer.py                ← AMP Mixed Precision + Resume + TensorBoard
│   ├── callbacks.py              ← Re-export EarlyStopping & ModelCheckpoint
│   └── metrics.py                ← Re-export metrics + Weighted F1
├── evaluation/
│   ├── evaluator.py              ← Test Evaluation + Classification Report
│   ├── roc.py                    ← One-vs-Rest ROC & AUC Curves
│   ├── misclassified.py         ← Misclassified Sample Extraction & Plotting
│   └── benchmark.py              ← Throughput (FPS) & Latency (p50/p95/p99)
├── reports/
│   ├── clinical_report.py        ← Structured JSON Clinical Diagnostic Report
│   └── performance_report.py     ← Markdown Performance Summary
├── inference/
│   ├── predict_brain_mri.py      ← Single-Image Predictor
│   └── batch_predict.py          ← Batch Directory Predictor
└── export/
    ├── onnx_export.py            ← ONNX Exporter (dynamo=False) & Verification
    └── torchscript_export.py     ← TorchScript Exporter (torch.jit.trace)
```

---

## 4. Key CLI Commands

```bash
# 1. Full Pipeline Execution
python run_brain_mri.py

# 2. Quick Sample Dry-Run
python run_brain_mri.py --sample --sample-size 50 --epochs 1

# 3. Resume Training from Checkpoint
python run_brain_mri.py --resume

# 4. Predict Single Image Disease & Generate Clinical Report
python predict_brain_mri.py --image processed_images/brain_mri/glioma/MRI_000001.png
```
