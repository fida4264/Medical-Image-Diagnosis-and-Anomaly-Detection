# PROPOSED_METHODOLOGY.md — Technical Methodology

---

## 1. System Architecture

The platform implements a layered, modular architecture comprising six functional tiers:

1. **Data Ingestion & Preprocessing Layer**: Per-modality preprocessing pipelines applying resizing (224×224 px), channel normalization using ImageNet statistics (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]), and tensor conversion.
2. **Modality Classification Layer**: A multi-class CNN classifier routing input images to the appropriate specialist predictor based on visual feature signatures.
3. **Disease Prediction Layer**: Seven specialist CNN predictors, each fine-tuned via transfer learning from ImageNet-pretrained backbones.
4. **Explainability Layer**: Plugin-based GradCAM engine producing class-discriminative localization maps.
5. **Anomaly Detection Layer**: Multi-strategy OOD engine computing composite anomaly scores.
6. **Report & Persistence Layer**: Structured report generation (PDF, Markdown, JSON) and SQLite audit trail.

---

## 2. Transfer Learning Strategy

Each specialist predictor implements fine-tuned transfer learning using the following backbone selection strategy:

| Modality | Backbone Architecture | Pretrained Dataset | Output Classes |
|---|---|---|---|
| Brain MRI | ResNet-50 / EfficientNet-B0 | ImageNet | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| Brain CT | ResNet-50 | ImageNet | 4 (Normal, Hemorrhage, Ischemic, Contusion) |
| Chest X-Ray | DenseNet-121 / ResNet-50 | ImageNet | 4 (Normal, Pneumonia, COVID-19, Tuberculosis) |
| Retina | EfficientNet-B0 | ImageNet | 5 (No DR, Mild, Moderate, Severe, Proliferative) |
| Histopathology | ResNet-50 | ImageNet | 8 (Adenocarcinoma, Squamous Cell, etc.) |
| Skin Lesion | EfficientNet-B0 | ImageNet | 9 (Melanoma, BCC, SCC, Nevi, etc.) |
| Ultrasound | ResNet-34 | ImageNet | 3 (Benign, Malignant, Normal) |

**Training Protocol:** Backbone layers frozen for initial epochs (feature extraction), followed by selective unfreezing (fine-tuning) of top layers with reduced learning rate (1e-4). Loss function: Categorical Cross-Entropy with label smoothing (α=0.1). Optimizer: Adam with cosine annealing schedule.

---

## 3. Gradient-weighted Class Activation Mapping (GradCAM)

For a CNN with convolutional feature maps A^k of spatial dimensions (u, v) for class c, the GradCAM saliency map is computed as:

**Step 1 — Compute gradient weights:**
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

where Z = u × v is the number of spatial locations.

**Step 2 — Generate weighted activation map:**
$$L_{GradCAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

**Step 3 — Postprocessing:**
- Bilinear upsampling to input image dimensions (224×224 px)
- Min-max normalization to [0,1] range
- Jet colormap application and alpha blending with original image
- Base64 PNG encoding for API transmission

---

## 4. Out-of-Distribution (OOD) Anomaly Detection

The platform implements a multi-strategy OOD detection framework with configurable fallback priority:

**Strategy 1 — Mahalanobis Distance (Primary):**
$$M(x) = \sqrt{(f(x) - \mu_c)^T \Sigma^{-1} (f(x) - \mu_c)}$$

where f(x) is the penultimate-layer feature embedding, μ_c is the class centroid, and Σ is the tied covariance matrix estimated on training data.

**Strategy 2 — Autoencoder Reconstruction Error (Fallback):**
$$r(x) = \|x - \hat{x}\|_2^2$$

where x̂ is the autoencoder reconstruction. High reconstruction error indicates OOD status.

**Strategy 3 — Maximum Softmax Confidence Threshold:**
$$P_{OOD}(x) = 1 - \max_c P(y=c|x)$$

Low maximum softmax probability indicates distributional uncertainty.

**Decision Fusion:** Composite anomaly score is computed as weighted average across available strategies. Risk tiers are assigned using clinical thresholds:

| Risk Level | Anomaly Score Range | Clinical Implication |
|---|---|---|
| LOW | 0.00 – 0.25 | Normal finding, high model confidence |
| MEDIUM | 0.25 – 0.50 | Borderline, recommend clinical review |
| HIGH | 0.50 – 0.75 | Significant anomaly, specialist review required |
| CRITICAL | 0.75 – 1.00 | Urgent clinical attention required |

---

## 5. Medical Report Generation

Structured medical reports are assembled from prediction results and formatted in three output modalities:

- **JSON** (structured clinical data, machine-readable)
- **Markdown** (human-readable, suitable for EHR attachments)
- **PDF** (regulatory-grade formal report via reportlab)

Each report contains: patient identifier, imaging modality, timestamp, predicted condition, confidence score, GradCAM heatmap reference, anomaly score, risk level, and clinician recommendations.
