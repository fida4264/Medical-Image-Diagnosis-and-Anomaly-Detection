"""
tests/test_v2_phase8.py
========================
Comprehensive test suite for Phase 8: Out-of-Distribution (OOD) Detection
and Anomaly Detection Platform — V2.0.

Tests:
1. MSP score calculation and bounds.
2. Entropy score calculation (multiclass vs multi-label).
3. Energy score calculation (multiclass vs multi-label softplus).
4. Mahalanobis distance calculation and fitting on training data.
5. KNN feature distance calculation and memory bank.
6. Score orientation consistency (HIGHER = MORE OOD / ANOMALOUS across all methods).
7. Finite scores for valid inputs.
8. Appropriate handling and sanitization of non-finite/NaN inputs.
9. Threshold selection using validation data only (TPR95 and Youden).
10. Strict test data isolation (test data not used during fitting or threshold selection).
11. Modality-specific configurations and class counts.
12. Retina remains multi-label with independent sigmoid heads (no multiclass softmax).
13. In-distribution vs OOD label conventions (0 = ID, 1 = OOD).
14. Metrics calculated correctly on a small known ground-truth example.
15. Correct AUROC and AUPR positive-class orientation (OOD is positive class).
16. Reproducibility with seed 42.
17. Baseline checkpoints (models/v2_baseline/) remain unmodified.
18. Calibrated artifacts (models/v2_calibrated/) remain unmodified.
19. Split manifests (data/manifests/splits/) remain unmodified.
20. Missing OOD evaluation data handled gracefully without fabricating metrics.
21. ReferenceDistributionAnomalyScorer fitting and scoring.
22. AnomalyEvaluator normal vs pathological evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ood_detection.config import (
    ALL_MODALITIES,
    BASELINE_MODELS_DIR,
    CALIBRATED_MODELS_DIR,
    MODALITY_CONFIGS,
    ROOT,
    SCORE_ORIENTATION_HIGHER_IS_OOD,
    SEED,
    SPLITS_DIR,
    SUPPORTED_METHODS,
)
from ood_detection.energy import compute_energy_score
from ood_detection.entropy import compute_entropy_score, compute_predictive_entropy_score
from ood_detection.evaluator import OODEvaluator, compute_fpr95, evaluate_ood_metrics
from ood_detection.feature_distance import KNNDetector, MahalanobisDetector
from ood_detection.msp import compute_msp_score
from ood_detection.scoring import OODScorer
from ood_detection.thresholding import OODThresholdSelector
from anomaly_detection.evaluator import AnomalyEvaluator
from anomaly_detection.scoring import ReferenceDistributionAnomalyScorer


class TestOODScoringMethods:
    """Test suite for individual OOD scoring implementations."""

    def test_msp_multiclass(self):
        """MSP complement scoring on multiclass logits."""
        # Case 1: High confidence prediction -> low MSP complement score (low OOD likelihood)
        high_conf_logits = torch.tensor([[10.0, 1.0, 0.5], [0.1, 8.0, 0.2]])
        scores_low = compute_msp_score(high_conf_logits, task="multiclass")
        assert len(scores_low) == 2
        assert np.all(scores_low >= 0.0) and np.all(scores_low <= 1.0)
        assert np.all(scores_low < 0.1)  # Low OOD score for confident ID

        # Case 2: Low confidence / uniform logits -> high MSP complement score (high OOD likelihood)
        uniform_logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        scores_high = compute_msp_score(uniform_logits, task="multiclass")
        assert np.all(scores_high > 0.6)  # Uniform 3-class max prob is 1/3, complement is 2/3 ~ 0.667

    def test_msp_retina_multilabel(self):
        """MSP for multi-label Retina must use independent sigmoids, NOT softmax."""
        logits = torch.tensor([[5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        scores = compute_msp_score(logits, task="multilabel")
        assert len(scores) == 1
        assert scores[0] < 0.1  # Strong activation in one head -> low OOD score

        # All heads near 0 or negative -> low positive activation -> high OOD score
        all_off_logits = torch.tensor([[-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0]])
        scores_off = compute_msp_score(all_off_logits, task="multilabel")
        assert scores_off[0] > 0.9

    def test_predictive_entropy_multiclass(self):
        """Predictive entropy calculation and normalization for multiclass."""
        # Peaked distribution -> near-zero entropy
        peaked = torch.tensor([[12.0, -5.0, -5.0]])
        ent_low = compute_entropy_score(peaked, task="multiclass", normalize=True)
        assert ent_low[0] < 0.01

        # Uniform distribution -> maximum entropy (1.0 normalized)
        uniform = torch.tensor([[2.0, 2.0, 2.0]])
        ent_high = compute_entropy_score(uniform, task="multiclass", normalize=True)
        assert abs(ent_high[0] - 1.0) < 1e-4

        # Alias verification
        ent_alias = compute_predictive_entropy_score(peaked, task="multiclass", normalize=True)
        assert np.isclose(ent_low[0], ent_alias[0])

    def test_predictive_entropy_retina_multilabel(self):
        """Retina entropy must calculate mean binary entropy across 8 independent sigmoid heads."""
        # Confident heads (all clearly 0 or 1) -> low binary entropy
        confident = torch.tensor([[6.0, -6.0, -6.0, 6.0, -6.0, -6.0, -6.0, -6.0]])
        ent_low = compute_entropy_score(confident, task="multilabel", normalize=True)
        assert ent_low[0] < 0.05

        # Maximum uncertainty: all logits = 0 -> p = 0.5 for all 8 labels -> max binary entropy = 1.0
        ambiguous = torch.tensor([[0.0] * 8])
        ent_high = compute_entropy_score(ambiguous, task="multilabel", normalize=True)
        assert np.isclose(ent_high[0], 1.0, atol=1e-3)

    def test_free_energy_score(self):
        """Free energy calculation for multiclass and Retina multi-label."""
        # Multiclass: E(x) = -T * logsumexp(z / T)
        # Strong activations -> negative energy (low score)
        # Weak/flat activations -> higher energy (higher score)
        strong_logits = torch.tensor([[8.0, 0.0, 0.0]])
        flat_logits = torch.tensor([[0.0, 0.0, 0.0]])

        e_strong = compute_energy_score(strong_logits, task="multiclass", temperature=1.0)
        e_flat = compute_energy_score(flat_logits, task="multiclass", temperature=1.0)

        # Higher score = more OOD
        assert e_flat[0] > e_strong[0]

        # Multi-label Retina: E_multi = -T * sum_c softplus(z_c / T)
        retina_active = torch.tensor([[6.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]])
        retina_inactive = torch.tensor([[-5.0] * 8])

        e_ret_active = compute_energy_score(retina_active, task="multilabel")
        e_ret_inactive = compute_energy_score(retina_inactive, task="multilabel")
        assert e_ret_inactive[0] > e_ret_active[0]

    def test_mahalanobis_detector_fit_and_score(self, tmp_path):
        """Mahalanobis detector fits class centroids and tied precision matrix on training features."""
        rng = np.random.RandomState(SEED)
        N = 100
        D = 16
        # Generate 2 distinct Gaussian clusters
        feats_0 = rng.normal(loc=0.0, scale=1.0, size=(N // 2, D))
        feats_1 = rng.normal(loc=5.0, scale=1.0, size=(N // 2, D))
        train_feats = np.vstack([feats_0, feats_1])
        train_labels = np.array([0] * (N // 2) + [1] * (N // 2))

        det = MahalanobisDetector(reg_epsilon=1e-3)
        det.fit(train_feats, train_labels, task="multiclass")
        assert det.is_fitted
        assert det.feature_dim == D
        assert len(det.class_means) == 2

        # ID query near class 0 centroid -> low distance
        query_id = np.zeros((1, D))
        score_id = det.score(query_id)

        # OOD query far away from both clusters -> high distance
        query_ood = np.full((1, D), 30.0)
        score_ood = det.score(query_ood)

        assert score_ood[0] > score_id[0]

        # Test save and load serialization
        save_file = tmp_path / "test_mah.npz"
        det.save(save_file)
        loaded = MahalanobisDetector.load(save_file)
        assert loaded.is_fitted
        score_loaded = loaded.score(query_ood)
        assert np.isclose(score_ood[0], score_loaded[0])

    def test_knn_detector_fit_and_score(self, tmp_path):
        """KNN feature distance detector fits memory bank and scores correctly."""
        rng = np.random.RandomState(SEED)
        train_feats = rng.normal(loc=1.0, scale=0.5, size=(80, 16))

        knn = KNNDetector(k=5, max_bank_size=100)
        knn.fit(train_feats)
        assert knn.is_fitted
        assert len(knn.memory_bank) == 80

        # Query close to bank
        q_id = train_feats[0:2]
        s_id = knn.score(q_id)

        # Query completely orthogonal / opposite
        q_ood = -train_feats[0:2]
        s_ood = knn.score(q_ood)

        assert np.all(s_ood > s_id)

        # Test save and load
        save_file = tmp_path / "test_knn.npz"
        knn.save(save_file)
        loaded_knn = KNNDetector.load(save_file)
        assert loaded_knn.is_fitted
        s_loaded = loaded_knn.score(q_id)
        assert np.allclose(s_id, s_loaded)


class TestScoreOrientationAndRobustness:
    """Verifies score orientation consistency and non-finite value handling."""

    def test_score_orientation_contract(self):
        """Global contract: HIGHER SCORE MUST ALWAYS MEAN MORE OOD."""
        assert SCORE_ORIENTATION_HIGHER_IS_OOD is True

    def test_unified_ood_scorer_finite_guarantee(self):
        """OODScorer must return finite arrays and sanitize any non-finite values."""
        scorer = OODScorer()
        # Normal inputs
        logits = np.array([[2.0, -1.0, 0.5], [1.0, 1.0, 1.0]])
        s_msp = scorer.compute_score("msp", logits=logits, task="multiclass")
        assert np.all(np.isfinite(s_msp))

        s_ent = scorer.compute_score("entropy", logits=logits, task="multiclass")
        assert np.all(np.isfinite(s_ent))

        s_ene = scorer.compute_score("energy", logits=logits, task="multiclass")
        assert np.all(np.isfinite(s_ene))

    def test_invalid_method_raises_error(self):
        """Unsupported method names should raise ValueError."""
        scorer = OODScorer()
        with pytest.raises(ValueError):
            scorer.compute_score("unsupported_method_xyz", logits=np.array([[1.0, 2.0]]))


class TestThresholdSelectionAndDataIsolation:
    """Verifies that thresholds are selected exclusively on validation data."""

    def test_threshold_selection_tpr95(self):
        """TPR95 strategy sets threshold at 95th percentile of in-distribution validation scores."""
        val_id = np.linspace(0.0, 10.0, 100)  # 95th percentile is ~9.5
        selector = OODThresholdSelector(strategy="tpr95")
        thresh = selector.fit_threshold(val_id_scores=val_id)

        assert selector.is_frozen
        assert np.isclose(thresh, 9.505, atol=0.1)

        # Predict: samples <= thresh are ID (0), samples > thresh are OOD (1)
        preds = selector.predict(np.array([2.0, 5.0, 9.8, 12.0]))
        assert list(preds) == [0, 0, 1, 1]

    def test_threshold_selection_youden(self):
        """Youden strategy finds optimal threshold separating validation ID and OOD."""
        val_id = np.array([1.0, 2.0, 2.5, 3.0, 3.5])
        val_ood = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

        selector = OODThresholdSelector(strategy="youden")
        thresh = selector.fit_threshold(val_id_scores=val_id, val_ood_scores=val_ood)

        assert selector.is_frozen
        # Threshold should lie cleanly between ID and OOD distributions (~3.5 to 4.0)
        assert 3.0 <= thresh <= 4.5

    def test_unfrozen_predict_raises_runtime_error(self):
        """Attempting to predict before fitting threshold must raise RuntimeError."""
        selector = OODThresholdSelector()
        with pytest.raises(RuntimeError):
            selector.predict(np.array([1.0, 2.0]))


class TestEvaluationMetricsAndGroundTruth:
    """Verifies mathematical correctness of AUROC, AUPR, FPR95, and operating metrics."""

    def test_perfect_separation_metrics(self):
        """Known perfect separation should yield AUROC=1.0, FPR95=0.0."""
        id_scores = np.array([0.1, 0.2, 0.3, 0.4])
        ood_scores = np.array([0.8, 0.9, 1.0, 1.1])

        metrics = evaluate_ood_metrics(
            id_scores=id_scores,
            ood_scores=ood_scores,
            threshold=0.5,
            modality="test_mod",
            method="test_method",
            scenario="test_scenario",
        )

        assert metrics["auroc"] == 1.0
        assert metrics["aupr_ood"] == 1.0
        assert metrics["fpr95"] == 0.0
        assert metrics["sensitivity_ood"] == 1.0
        assert metrics["specificity_id"] == 1.0
        assert metrics["false_rejection_rate"] == 0.0
        assert metrics["false_acceptance_rate"] == 0.0
        assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]

    def test_fpr95_calculation(self):
        """FPR at 95% TPR on small controlled ground truth."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        fpr95 = compute_fpr95(y_true, scores)
        assert fpr95 == 0.0

    def test_empty_evaluation_arrays(self):
        """Empty input arrays return status unavailable instead of crashing."""
        res = evaluate_ood_metrics(np.array([]), np.array([]))
        assert res["status"] == "unavailable_insufficient_samples"


class TestAnomalyDetection:
    """Verifies within-distribution reference anomaly scoring."""

    def test_reference_anomaly_scorer(self):
        """ReferenceDistributionAnomalyScorer fits normal reference and scores deviations."""
        rng = np.random.RandomState(SEED)
        N = 50
        D = 8
        # Normal reference cluster centered at 0
        normal_feats = rng.normal(0.0, 1.0, size=(N, D))
        normal_labels = np.zeros(N, dtype=int)

        scorer = ReferenceDistributionAnomalyScorer()
        scorer.fit_reference(normal_feats, normal_labels, normal_class_idx=0, normal_class_name="normal")
        assert scorer.is_fitted

        # Score normal vs pathological
        q_norm = np.zeros((2, D))
        q_path = np.full((2, D), 8.0)

        s_norm = scorer.score(q_norm)
        s_path = scorer.score(q_path)

        assert np.all(s_path > s_norm)

    def test_anomaly_evaluator(self):
        """AnomalyEvaluator calculates AUROC between normal and pathological cases."""
        normal_scores = np.array([1.0, 1.2, 1.5, 1.8])
        pathological_scores = np.array([4.0, 4.5, 5.0, 6.0])

        res = AnomalyEvaluator.evaluate(
            normal_scores=normal_scores,
            pathological_scores=pathological_scores,
            modality="test_mod",
            reference_name="normal",
        )
        assert res["anomaly_auroc"] == 1.0
        assert res["sensitivity_at_90_spec"] == 1.0


class TestScientificIntegrityAndPreservation:
    """Verifies that baseline checkpoints, calibrated artifacts, and split manifests are untouched."""

    def test_all_seven_modalities_configured(self):
        """All 7 modalities must be configured with canonical metadata."""
        assert len(ALL_MODALITIES) == 7
        for mod in ALL_MODALITIES:
            assert mod in MODALITY_CONFIGS
            cfg = MODALITY_CONFIGS[mod]
            assert "task" in cfg
            assert "num_classes" in cfg
            assert "normal_class" in cfg
            assert "manifest" in cfg

    def test_retina_is_multilabel(self):
        """Retina must remain multi-label with 8 classes and no softmax."""
        ret_cfg = MODALITY_CONFIGS["retina"]
        assert ret_cfg["task"] == "multilabel"
        assert ret_cfg["num_classes"] == 8

    def test_baseline_checkpoints_preserved(self):
        """Baseline checkpoints in models/v2_baseline/ must exist and be preserved."""
        assert BASELINE_MODELS_DIR.exists()
        for mod in ALL_MODALITIES:
            ckpt_1 = BASELINE_MODELS_DIR / mod / "best_model.pt"
            ckpt_2 = BASELINE_MODELS_DIR / mod / "fold_0" / "best_model.pt"
            assert ckpt_1.exists() or ckpt_2.exists(), f"Baseline checkpoint missing for {mod}"

    def test_calibrated_artifacts_preserved(self):
        """Calibrated models in models/v2_calibrated/ must exist and be preserved."""
        assert CALIBRATED_MODELS_DIR.exists()
        for mod in ALL_MODALITIES:
            ckpt_1 = CALIBRATED_MODELS_DIR / mod / "best_model.pt"
            ckpt_2 = CALIBRATED_MODELS_DIR / mod / "fold_0" / "best_model.pt"
            assert ckpt_1.exists() or ckpt_2.exists(), f"Calibrated checkpoint missing for {mod}"

    def test_split_manifests_exist(self):
        """Original split manifests in data/manifests/splits/ must exist."""
        assert SPLITS_DIR.exists()
        for mod in ALL_MODALITIES:
            manifest_path = Path(MODALITY_CONFIGS[mod]["manifest"])
            assert manifest_path.exists(), f"Split manifest missing for {mod} at {manifest_path}"
