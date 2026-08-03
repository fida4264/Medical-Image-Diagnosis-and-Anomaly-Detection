# Module 6 Design Specification — Medical Image Modality Classification Framework

---

## 1. Executive Summary

Module 6 serves as the **Intelligent Routing Engine** for the Medical AI System. It automatically identifies the imaging modality of uploaded medical images without reliance on user manual tagging or file path hints.

### Target Modalities
1. **Brain MRI** (`brain_mri`) → Routes to Module 7 (Brain MRI Anomaly Detection & Segmentation)
2. **Brain CT** (`brain_ct`) → Routes to Module 8 (Brain CT Scan Classification)
3. **Chest X-Ray** (`chest_xray`) → Routes to Module 9 (Chest X-Ray Disease Classification)
4. **Histopathology** (`histopathology`) → Routes to Module 10 (Histopathology Cancer Subtyping)
5. **Retina** (`retina`) → Routes to Module 11 (Retina Diabetic Retinopathy Classification)
6. **Skin Lesion** (`skin`) → Routes to Module 12 (Skin Lesion Cancer Classification)
7. **Ultrasound** (`ultrasound`) → Routes to Module 13 (Ultrasound Breast Anomaly Detection)

---

## 2. Architectural Design

```
Raw/Preprocessed Image Input (224x224x3)
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ PyTorch Transfer Learning Backbone               │
│ (MobileNetV3-Small / ResNet18 / EfficientNet-B0) │
└──────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ Global Adaptive Average Pooling                  │
└──────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ BatchNorm1D + Dropout (p=0.3) + Linear Softmax   │
└──────────────────────────────────────────────────┘
                   │
                   ▼
       Modality Softmax Logits
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
Top-K Prediction Scores    Routing Target Module ID
```

---

## 3. Directory Structure

```
modality_classifier/
├── __init__.py
├── config.py                  ← DataClass Configuration Loader
├── datasets/
│   ├── __init__.py
│   ├── dataset_builder.py     ← Stratified Train/Val/Test Split Builder
│   └── data_loader.py         ← PyTorch Dataset & DataLoader
├── models/
│   ├── __init__.py
│   ├── base_model.py          ← ModalityClassifier PyTorch Module
│   ├── backbone_factory.py    ← Transfer Learning Backbone Factory
│   └── classifier_head.py     ← Modular Head
├── training/
│   ├── __init__.py
│   ├── trainer.py             ← PyTorch Training Loop
│   ├── metrics.py             ← Accuracy, Precision, Recall, Macro F1
│   └── callbacks.py           ← Early Stopping & Checkpointing
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py           ← Test Evaluation & Heatmap Plotting
├── inference/
│   ├── __init__.py
│   ├── predictor.py           ← High-performance Predictor & Routing
│   └── onnx_exporter.py       ← PyTorch to ONNX Exporter
└── utils/
    ├── __init__.py
    ├── logging.py
    └── visualization.py
```

---

## 4. Key CLI Commands

```bash
# 1. Train Modality Classifier Model
python train_modality_classifier.py --sample --sample-size 50 --epochs 2

# 2. Evaluate Trained Model on Test Set
python evaluate_modality_classifier.py

# 3. Predict Single Image Modality & Routing
python predict_modality.py --image processed_images/chest_xray/normal/XRAY_000001.png

# 4. Export PyTorch Checkpoint to ONNX
python export_modality_onnx.py
```
