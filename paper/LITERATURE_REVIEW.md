# LITERATURE_REVIEW.md — Related Work & Comparative Analysis

---

## 1. Overview

The development of AI-assisted medical image diagnosis has accelerated substantially following the success of deep convolutional neural networks (CNNs) in the ImageNet challenge (Krizhevsky et al., 2012). This section reviews the key related works across the clinical domains addressed by this platform, identifies the limitations of existing approaches, and positions this platform's contributions within the field.

---

## 2. Deep Learning for Medical Image Classification

### 2.1 Chest X-Ray Analysis
- **CheXNet (Rajpurkar et al., 2017)**: 121-layer DenseNet trained on 112,000 chest X-rays achieving radiologist-level pneumonia detection on the ChestX-ray14 dataset. Limitation: single modality; no OOD detection or deployment path.
- **CheXpert (Irvin et al., 2019)**: Multi-label chest condition classifier with uncertainty labels. Limitation: no explainability pipeline, no clinical report generation.

### 2.2 Retinal Fundus Analysis
- **Gulshan et al. (Google DeepMind, 2016)**: Deep learning algorithm for diabetic retinopathy detection achieving AUC 0.99 on ophthalmologist-graded sets. Limitation: single condition, closed implementation.
- **DeepDR (Liu et al., 2019)**: Multi-task diabetic retinopathy classification network. Limitation: no anomaly detection or OOD flagging.

### 2.3 Skin Lesion Analysis
- **Esteva et al. (2017)**: CNN-based classification of 129,450 clinical dermoscopy images matching dermatologist accuracy. Limitation: binary classification only, no GradCAM integration.
- **ISIC 2019 Challenge Winners**: EfficientNet-based multi-class lesion classifiers. Limitation: research-only, no production REST API or deployment infrastructure.

### 2.4 Histopathology
- **Campanella et al. (2019)**: Weakly-supervised cancer detection on 44,732 whole-slide pathology images. Limitation: whole-slide resolution, not suitable for clinical patch classification in resource-constrained settings.

### 2.5 Brain MRI & CT
- **Pereira et al. (2016)**: CNN for brain tumor segmentation via multimodal MRI. Limitation: segmentation focus, no GradCAM XAI or risk stratification.

---

## 3. Explainable AI in Medical Imaging

- **Grad-CAM (Selvaraju et al., 2017)**: Gradient-weighted Class Activation Mapping providing visual explanations for any CNN-based classifier. This platform extends Grad-CAM across all 7 modalities through a plugin-based CAM Registry.
- **LIME (Ribeiro et al., 2016)**: Local Interpretable Model-agnostic Explanations. Not selected due to higher computational overhead and lower spatial precision compared to GradCAM.

---

## 4. Out-of-Distribution Detection

- **Mahalanobis Distance (Lee et al., 2018)**: Confidence scoring using Mahalanobis distance from class-conditional Gaussian distributions in feature space. Adopted as primary OOD strategy.
- **ODIN (Liang et al., 2018)**: Temperature scaling and input perturbation for OOD detection. Incorporated as fallback strategy.
- **Deep SVDD (Ruff et al., 2018)**: One-class anomaly detection in embedded space. Included as optional deep anomaly strategy.

---

## 5. Comparative Summary Table

| Work / System | Modalities | Explainability | OOD Detection | REST API | Report Generation | Production Deployment |
|---|---|---|---|---|---|---|
| CheXNet (2017) | 1 (CXR) | ❌ | ❌ | ❌ | ❌ | ❌ |
| DeepDR (2019) | 1 (Retina) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Esteva et al. (2017) | 1 (Skin) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Grad-CAM framework | Model-agnostic | ✅ | ❌ | ❌ | ❌ | ❌ |
| Mahalanobis OOD | Model-agnostic | ❌ | ✅ | ❌ | ❌ | ❌ |
| **This Platform (2026)** | **7 Modalities** | **✅ GradCAM** | **✅ Multi-strategy** | **✅ FastAPI** | **✅ PDF/MD/JSON** | **✅ Docker + K8s** |

---

## 6. Positioning Statement

This platform is, to the authors' knowledge, the first open-source system to integrate **multi-modality medical image classification**, **GradCAM explainability**, **multi-strategy OOD anomaly detection**, **audit-grade database logging**, **automated multi-format medical report generation**, and **production-ready containerized deployment** within a single, orchestrated framework.
