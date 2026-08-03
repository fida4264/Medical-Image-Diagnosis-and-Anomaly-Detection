# Module 16.4 — Unknown Disease Detection Subsystem

**Medical Image Diagnosis and Anomaly Detection Platform**  
**Phase 5 — Advanced AI & Anomaly Detection**  
**Status:** ✅ COMPLETE · 89/89 Anomaly Tests Passed (264 Total Platform Tests Passed)

---

## 1. Overview

Module 16.4 implements the **Unknown Disease Detection** subsystem. It determines whether an input sample belongs to an **unseen / unknown disease category** by comparing its feature embedding to known disease class prototypes.

This subsystem complements confidence and OOD detection by providing vector-space prototype matching.

---

## 2. Architecture & Pipeline

```
PredictionResult
      │
      ▼
1. FeatureEmbeddingManager  ────── Extract / Generate (512-d normalized vector)
      │
      ▼
2. PrototypeManager         ────── Retrieve / Compute Class Centroids (μ_c)
      │
      ▼
3. SimilarityEngine         ────── Cosine Similarity / Euclidean Distance to Prototypes
      │
      ▼
4. NoveltyEstimator         ────── Calibrate Novelty Score via α·(1−sim) + β·(1−conf)
      │
      ▼
5. UnknownDiseaseDetector   ────── Threshold Decision → KNOWN vs UNKNOWN
      │
      ▼
AnomalyResult Enrichment    ────── Populate known_unknown_status, novelty_score, etc.
```

---

## 3. Key Components

| Module | Class | Description |
|---|---|---|
| `feature_embedding_manager.py` | `FeatureEmbeddingManager` | Extracts 512-d feature vector from result, model adapter, or deterministic hash stub. |
| `prototype_manager.py` | `PrototypeManager` | Computes L2-normalized mean centroids for each in-distribution class. |
| `similarity_engine.py` | `SimilarityEngine` | Computes Cosine similarity, Euclidean distance, or Dot product across prototypes. |
| `novelty_estimator.py` | `NoveltyEstimator` | Fuses prototype similarity (70%) and prediction confidence (30%) with sigmoid/tanh calibration. |
| `unknown_detector.py` | `UnknownDiseaseDetector` | Orchestrates pipeline, evaluates modality thresholds, and constructs `AnomalyResult`. |

---

## 4. Prototype Strategy & Similarity Metrics

### Prototype Centroid Calculation
$$\mu_c = \frac{1}{N_c} \sum_{i=1}^{N_c} z_i, \quad \hat{\mu}_c = \frac{\mu_c}{\|\mu_c\|_2}$$

### Similarity Metrics Supported
1. **Cosine Similarity**: $\text{sim}(q, \hat{\mu}_c) = \frac{1 + q \cdot \hat{\mu}_c}{2} \in [0, 1]$
2. **Euclidean Similarity**: $\text{sim}(q, \hat{\mu}_c) = \frac{1}{1 + \|q - \hat{\mu}_c\|_2} \in (0, 1]$
3. **Dot Product**: $q \cdot \hat{\mu}_c$

---

## 5. Threshold Configuration

`UnknownConfiguration` manages novelty thresholds:
- Default `novelty_threshold`: `0.60`
- Default `similarity_threshold`: `0.70`
- Per-modality overrides:
  - `brain_mri`: 0.58
  - `chest_xray`: 0.60
  - `histopathology`: 0.55
  - `retina`: 0.60
  - `skin`: 0.62
  - `ultrasound`: 0.65

---

## 6. Output AnomalyResult Fields

- `status`: `AnomalyStatus.UNKNOWN_DISEASE` or `AnomalyStatus.KNOWN_DISEASE`
- `known_unknown_status`: `"UNKNOWN"` or `"KNOWN"`
- `risk_level`: `RiskLevel.HIGH` (unknown) or `RiskLevel.LOW` (known)
- `recommended_action`: `RecommendedAction.ESCALATE` (unknown) or `RecommendedAction.PROCEED` (known)
- `risk_score`: Calibrated novelty score in $[0.0, 1.0]$
- `unknown_disease_analysis`:
  - `is_unknown`: bool
  - `novelty_score`: float
  - `similarity_score`: float
  - `nearest_class`: str
  - `similarity_threshold`: float

---

## 7. Limitations

1. **Synthetic Prototypes in Test Environments**: Seeding synthetic prototypes is used when zero actual training embeddings are registered.
2. **Fixed Class Centroids**: Prototypes assume unimodal class distributions in embedding space. Multi-modal class clusters will be addressed in future open-set expansion modules.

---

## 8. Verification Results

```
============================= 89 passed in 0.72s ==============================
(89 anomaly tests + 175 explainability tests = 264 total platform tests)
```
