# Module 16.3 — Confidence & Out-of-Distribution (OOD) Detection

**Medical Image Diagnosis and Anomaly Detection Platform**  
**Phase 5 — Advanced AI & Anomaly Detection**  
**Status:** ✅ COMPLETE · 66/66 Anomaly Tests Passed (241 Total Platform Tests Passed)

---

## 1. Overview

Module 16.3 implements the first two production anomaly detection strategies:

| Strategy | Key | Algorithm |
|---|---|---|
| **Confidence Threshold Detection** | `confidence_threshold` | Multi-tier threshold comparison against per-modality confidence levels |
| **Mahalanobis Distance OOD Detection** | `mahalanobis` | Minimum Mahalanobis distance from in-distribution class mean prototypes |

Both strategies implement `IDetectionStrategy` and integrate seamlessly into the existing `AnomalyDetectionEngine` → `AnomalyRegistry` pipeline defined in Modules 16.1–16.2.

---

## 2. Package Structure

```
medical_ai_engine/
└── anomaly/
    └── strategies/
        ├── __init__.py                   ← Package exports
        ├── strategy_exceptions.py        ← Typed exception hierarchy
        ├── strategy_configuration.py     ← StrategyConfiguration & ModalityThresholds
        ├── feature_extractor.py          ← FeatureExtractor (model adapter / stub)
        ├── feature_statistics.py         ← FeatureStatistics (class means + precision matrix)
        ├── distance_calculator.py        ← DistanceCalculator (Mahalanobis, Euclidean, Cosine)
        ├── threshold_manager.py          ← ThresholdManager (per-modality lookup & tuning)
        ├── confidence_threshold.py       ← ConfidenceThresholdStrategy plugin
        └── mahalanobis_detector.py       ← MahalanobisOODStrategy plugin
```

---

## 3. Confidence Threshold Strategy

### Algorithm

Classifies prediction confidence into four tiers using per-modality thresholds:

| Level | Condition | Flag | Is Anomaly |
|---|---|---|---|
| `HIGH` | `confidence >= high_confidence` | `TRUSTED` | ❌ No |
| `MEDIUM` | `confidence >= medium_confidence` | `TRUSTED` | ❌ No |
| `LOW` | `confidence >= low_confidence` | `LOW_CONFIDENCE` | ✅ Yes |
| `REJECT` | `confidence < reject_threshold` | `CRITICAL` | ✅ Yes |

### Per-Modality Default Thresholds

| Modality | High | Medium | Low | Reject | OOD Threshold |
|---|---|---|---|---|---|
| `brain_mri` | 0.88 | 0.70 | 0.50 | 0.30 | 3.2 |
| `chest_xray` | 0.85 | 0.70 | 0.50 | 0.30 | 3.5 |
| `histopathology` | 0.90 | 0.70 | 0.50 | 0.30 | 3.0 |
| `retina` | 0.85 | 0.70 | 0.50 | 0.30 | 3.5 |
| `skin` | 0.82 | 0.70 | 0.50 | 0.30 | 3.8 |
| `ultrasound` | 0.80 | 0.70 | 0.50 | 0.30 | 4.0 |

---

## 4. Mahalanobis Distance OOD Detection

### Algorithm

```
D_M(z, μ_c) = √[(z − μ_c)ᵀ · Σ⁻¹ · (z − μ_c)]
OOD decision: min_c D_M(z, μ_c) > threshold
```

### Execution Steps

1. **Feature Extraction**: Extract penultimate feature vector `z ∈ ℝᴰ` from model adapter or synthetic stub.
2. **Class Mean Computation**: `μ_c = (1/N_c) Σᵢ zᵢ` for each class `c`.
3. **Regularized Covariance**: `Σ_reg = Σ + ε·I` (Tikhonov regularization, default `ε=1e-4`).
4. **Precision Matrix**: `Σ⁻¹ = pinv(Σ_reg)` via Moore-Penrose pseudoinverse.
5. **Distance Computation**: Minimum Mahalanobis distance to any class mean prototype.
6. **OOD Decision**: `is_ood = dist > threshold_modality`.

### AnomalyContext Outputs

| Field | Description |
|---|---|
| `context.feature_vector` | Extracted feature vector |
| `context.feature_distance` | Mahalanobis distance |
| `context.ood_analysis.is_ood` | Boolean OOD decision |
| `context.ood_analysis.ood_score` | Mahalanobis distance value |
| `context.ood_analysis.nearest_class` | Closest in-distribution class |

---

## 5. Performance Measurements

All strategies record timing metrics to `AnomalyContext.performance_metrics`:

| Metric Key | Description |
|---|---|
| `confidence_eval_time_ms` | Confidence threshold evaluation time |
| `feature_extraction_time_ms` | Feature vector extraction time |
| `distance_computation_time_ms` | Mahalanobis distance computation time |
| `mahalanobis_total_time_ms` | Total Mahalanobis strategy execution time |

---

## 6. Known Limitations

1. **Synthetic Feature Stubs**: Without a connected model adapter, `FeatureExtractor` generates deterministic random feature vectors. Real deployment requires model integration (Module 16.4+).
2. **Static Class Statistics**: `FeatureStatistics` is currently pre-fitted with synthetic class cluster statistics. Production deployment requires calibration from in-distribution training data.
3. **Tied Covariance**: A single shared precision matrix is used across all classes (tied Gaussian assumption). Class-specific covariances are a future enhancement.

---

## 7. Verification Results

```
============================= 66 passed in 0.64s ==============================
(66 anomaly tests + 175 explainability tests = 241 total platform tests)
```
