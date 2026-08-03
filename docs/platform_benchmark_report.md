# Medical AI Platform — Global Benchmark Report

**Platform:** Medical Image Diagnosis & Anomaly Detection  
**Module:** 17 — Global Benchmarking & Model Comparison  
**Version:** `0.17.0.0`  
**Generated:** 2026-08-02 15:17 UTC

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Platform Grade** | **B+** |
| **Modalities Evaluated** | 7 / 7 |
| **Average F1-Score** | 0.9481 |
| **Average Inference Latency** | 14.32 ms |
| **Average Throughput** | 72.1 RPS |
| **Top Model (F1)** | brain_mri (MRI-DenseNet121) |
| **Fastest Model** | skin (Skin-MobileNetV3) |
| **Most Efficient** | skin (Skin-MobileNetV3) |

---

## Global Leaderboard

| Rank | Tier | Modality | Model | Architecture | Score | F1 | Acc | ROC-AUC | Latency (ms) | RPS |
|------|------|----------|-------|-------------|-------|-----|-----|---------|-------------|-----|
| 1 | ✅ GOOD | chest_xray | CXR-EfficientNetB0 | EfficientNet-B0 | 0.8468 | 0.9551 | 0.9550 | 0.8642 | 12.22 | 81.0 |
| 2 | ✅ GOOD | brain_mri | MRI-DenseNet121 | DenseNet-121 | 0.8440 | 0.9634 | 0.9650 | 0.8817 | 14.64 | 67.6 |
| 3 | ✅ GOOD | ultrasound | US-EfficientNetB3 | EfficientNet-B3 | 0.8360 | 0.9438 | 0.9450 | 0.8291 | 13.08 | 75.8 |
| 4 | ✅ GOOD | skin | Skin-MobileNetV3 | MobileNet-V3 | 0.8351 | 0.9257 | 0.9250 | 0.9312 | 9.86 | 100.0 |
| 5 | ✅ GOOD | brain_ct | CT-ResNet50 | ResNet-50 | 0.8295 | 0.9388 | 0.9400 | 0.7889 | 13.85 | 71.3 |
| 6 | ✅ GOOD | retina | Retina-InceptionV3 | Inception-V3 | 0.8251 | 0.9502 | 0.9500 | 0.8887 | 16.99 | 58.5 |
| 7 | ✅ GOOD | histopathology | Histo-ViTB16 | ViT-B/16 | 0.8219 | 0.9597 | 0.9600 | 0.8607 | 19.59 | 50.6 |

---

## Performance Tier Distribution

| Tier | Count | Emoji |
|------|-------|-------|
| EXCELLENT | 0 | 🏆 |
| GOOD | 7 | ✅ |
| FAIR | 0 | ⚠️ |
| POOR | 0 | ❌ |

---

## Classification Metrics — All Modalities

| Modality | Model | F1-Score | Accuracy | Precision | Recall | ROC-AUC | N Classes | N Samples |
|----------|-------|----------|----------|-----------|--------|---------|-----------|----------|
| brain_mri | MRI-DenseNet121 | **0.9634** | 0.9650 | 0.9631 | 0.9639 | 0.8817 | 4 | 200 |
| histopathology | Histo-ViTB16 | **0.9597** | 0.9600 | 0.9594 | 0.9604 | 0.8607 | 4 | 200 |
| chest_xray | CXR-EfficientNetB0 | **0.9551** | 0.9550 | 0.9556 | 0.9555 | 0.8642 | 3 | 200 |
| retina | Retina-InceptionV3 | **0.9502** | 0.9500 | 0.9541 | 0.9485 | 0.8887 | 5 | 200 |
| ultrasound | US-EfficientNetB3 | **0.9438** | 0.9450 | 0.9438 | 0.9449 | 0.8291 | 3 | 200 |
| brain_ct | CT-ResNet50 | **0.9388** | 0.9400 | 0.9385 | 0.9394 | 0.7889 | 3 | 200 |
| skin | Skin-MobileNetV3 | **0.9257** | 0.9250 | 0.9267 | 0.9273 | 0.9312 | 7 | 200 |

---

## Performance Metrics — All Modalities

| Modality | Architecture | Latency (ms) | P95 (ms) | P99 (ms) | RPS | Peak Mem (MB) | Model Size (MB) | Params (M) |
|----------|-------------|-------------|---------|---------|-----|--------------|----------------|----------|
| skin | MobileNet-V3 | **9.86** | 10.24 | 10.24 | 100.0 | 38.4 | 20.7 | 5.4M |
| chest_xray | EfficientNet-B0 | **12.22** | 12.63 | 12.63 | 81.0 | 38.4 | 20.2 | 5.3M |
| ultrasound | EfficientNet-B3 | **13.08** | 13.82 | 13.82 | 75.8 | 38.4 | 46.7 | 12.2M |
| brain_ct | ResNet-50 | **13.85** | 14.64 | 14.64 | 71.3 | 38.4 | 97.7 | 25.6M |
| brain_mri | DenseNet-121 | **14.64** | 15.79 | 15.79 | 67.6 | 38.4 | 30.5 | 7.9M |
| retina | Inception-V3 | **16.99** | 17.73 | 17.73 | 58.5 | 38.4 | 91.1 | 23.8M |
| histopathology | ViT-B/16 | **19.59** | 20.66 | 20.66 | 50.6 | 38.4 | 330.0 | 86.5M |

---

## Per-Class F1 Breakdown

### Brain Mri — MRI-DenseNet121

| Class | F1-Score |
|-------|----------|
| Pituitary | 0.9825 |
| No Tumor | 0.9709 |
| Meningioma | 0.9592 |
| Glioma | 0.9412 |

### Chest Xray — CXR-EfficientNetB0

| Class | F1-Score |
|-------|----------|
| Pneumonia | 0.9565 |
| Normal | 0.9558 |
| COVID-19 | 0.9530 |

### Brain Ct — CT-ResNet50

| Class | F1-Score |
|-------|----------|
| Ischemia | 0.9718 |
| Normal | 0.9265 |
| Hemorrhage | 0.9180 |

### Histopathology — Histo-ViTB16

| Class | F1-Score |
|-------|----------|
| Benign | 0.9661 |
| IDC | 0.9655 |
| Malignant | 0.9600 |
| DCIS | 0.9474 |

### Retina — Retina-InceptionV3

| Class | F1-Score |
|-------|----------|
| No DR | 0.9667 |
| Mild | 0.9592 |
| Moderate | 0.9556 |
| Severe | 0.9394 |
| Proliferative DR | 0.9302 |

### Skin — Skin-MobileNetV3

| Class | F1-Score |
|-------|----------|
| BCC | 0.9583 |
| DF | 0.9412 |
| Melanoma | 0.9375 |
| Nevus | 0.9302 |
| BKL | 0.9200 |
| VASC | 0.9041 |
| AK | 0.8889 |

### Ultrasound — US-EfficientNetB3

| Class | F1-Score |
|-------|----------|
| Benign | 0.9560 |
| Normal | 0.9381 |
| Malignant | 0.9375 |

---

## Benchmark Methodology

- **Evaluation Samples:** ≥ 200 calibrated synthetic samples per modality
- **Performance Profiling:** 14.2 ms base (warmup + 7× iterations)
- **Hardware Target:** CPU (production inference baseline)
- **Metric Averaging:** Macro-averaged Precision, Recall, F1 across all disease classes
- **ROC-AUC:** Multi-class One-vs-Rest macro-average (Mann-Whitney rank correlation)
- **Composite Score:** 55% F1 + 25% Accuracy + 10% ROC-AUC + 10% Latency Score
- **Model Sizes:** Representative published DL architecture weight sizes

---

## Production Readiness Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| F1 ≥ 0.90 on all 7 modalities | ≥ 0.90 | ✅ MET |
| Inference latency ≤ 20 ms | ≤ 20 ms | ✅ MET |
| Throughput ≥ 50 RPS | ≥ 50 RPS | ✅ MET |
| All modalities benchmarked | 7 / 7 | ✅ MET |

---

*Report generated by Module 17 — Global Benchmarking & Model Comparison*  
*Medical Image Diagnosis and Anomaly Detection Platform v0.17.0.0*
