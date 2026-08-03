"""
test_anomaly_strategies.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Unit test suite for confidence and Mahalanobis strategies.

Covers:
  - ConfidenceThresholdStrategy per-modality evaluation
  - FeatureExtractor extraction & stub fallback
  - FeatureStatistics class mean & precision matrix fitting
  - DistanceCalculator Mahalanobis, Euclidean, Cosine calculations
  - ThresholdManager per-modality lookups
  - MahalanobisOODStrategy end-to-end detection
  - Integration with AnomalyRegistry, ExecutionPipeline, & AnomalyDetectionEngine
  - Concurrency & multi-modality support
"""

from __future__ import annotations

import concurrent.futures
import numpy as np
import pytest

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.anomaly_detection_engine import AnomalyDetectionEngine
from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from medical_ai_engine.anomaly.framework.pipeline import ExecutionPipeline
from medical_ai_engine.anomaly.strategies import (
    ConfidenceThresholdStrategy,
    DistanceCalculator,
    FeatureExtractor,
    FeatureStatistics,
    MahalanobisOODStrategy,
    ModalityThresholds,
    StrategyConfiguration,
    ThresholdManager,
)


@pytest.fixture
def strategy_config() -> StrategyConfiguration:
    return StrategyConfiguration()


# ── T01: ConfidenceThresholdStrategy ──────────────────────────────────────────

class TestConfidenceThresholdStrategy:

    def test_high_confidence(self, strategy_config):
        strat = ConfidenceThresholdStrategy(strategy_config)
        strat.initialize({})
        ctx = AnomalyContext(request_id="r1", session_id="s1", modality="brain_mri", confidence=0.92)
        is_anomaly = strat.detect(ctx)
        assert is_anomaly is False
        assert ctx.confidence_analysis["confidence_level"] == "HIGH"
        assert ctx.confidence_analysis["confidence_flag"] == "TRUSTED"

    def test_low_confidence_flagged_as_anomaly(self, strategy_config):
        strat = ConfidenceThresholdStrategy(strategy_config)
        strat.initialize({})
        ctx = AnomalyContext(request_id="r2", session_id="s2", modality="brain_mri", confidence=0.55)
        is_anomaly = strat.detect(ctx)
        assert is_anomaly is True
        assert ctx.confidence_analysis["confidence_level"] == "LOW"

    def test_per_modality_thresholds(self, strategy_config):
        strat = ConfidenceThresholdStrategy(strategy_config)
        strat.initialize({})
        # Histopathology has high threshold (0.90)
        ctx = AnomalyContext(request_id="r3", session_id="s3", modality="histopathology", confidence=0.88)
        strat.detect(ctx)
        assert ctx.confidence_analysis["confidence_level"] == "MEDIUM"


# ── T02: FeatureExtractor & FeatureStatistics ─────────────────────────────────

class TestFeatureComponents:

    def test_feature_extractor(self):
        extractor = FeatureExtractor(feature_dim=256)
        feats = extractor.extract_features(model_adapter=None, image_input=None)
        assert isinstance(feats, np.ndarray)
        assert feats.shape == (256,)

    def test_feature_statistics_fit(self):
        stats = FeatureStatistics()
        sample_data = {
            "class_a": np.random.randn(20, 128).astype(np.float32),
            "class_b": np.random.randn(20, 128).astype(np.float32) + 2.0,
        }
        stats.fit(sample_data)
        assert stats.is_fitted is True
        assert "class_a" in stats.class_means
        assert stats.precision_matrix.shape == (128, 128)


# ── T03: DistanceCalculator ───────────────────────────────────────────────────

class TestDistanceCalculator:

    def test_mahalanobis_distance(self):
        stats = FeatureStatistics()
        stats.fit_synthetic_defaults(["cls_1", "cls_2"], feature_dim=64)
        sample_vec = stats.class_means["cls_1"] + 0.01  # very close to cls_1

        dist, nearest = DistanceCalculator.compute_mahalanobis(
            feature_vec=sample_vec,
            class_means=stats.class_means,
            precision_matrix=stats.precision_matrix,
        )
        assert dist >= 0.0
        assert nearest == "cls_1"

    def test_euclidean_distance(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 6.0, 8.0])
        dist = DistanceCalculator.compute_euclidean(v1, v2)
        assert pytest.approx(dist, 0.01) == 7.071

    def test_cosine_distance(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        dist = DistanceCalculator.compute_cosine_distance(v1, v2)
        assert pytest.approx(dist, 0.01) == 1.0


# ── T04: MahalanobisOODStrategy ─────────────────────────────────────────────

class TestMahalanobisOODStrategy:

    def test_mahalanobis_detection_flow(self, strategy_config):
        strat = MahalanobisOODStrategy(strategy_config)
        strat.initialize({})
        ctx = AnomalyContext(request_id="r_mah_1", session_id="s_mah_1", modality="brain_mri", confidence=0.90)

        # Detect
        is_ood = strat.detect(ctx)
        assert isinstance(is_ood, bool)
        assert ctx.ood_analysis is not None
        assert "ood_score" in ctx.ood_analysis
        assert ctx.feature_distance is not None

    def test_metadata_contract(self):
        strat = MahalanobisOODStrategy()
        strat.initialize({})
        meta = strat.get_metadata()
        assert meta["key"] == "mahalanobis"
        assert meta["requires_feature_vector"] is True


# ── T05: End-to-End Pipeline & Engine Integration ────────────────────────────

class TestStrategyEngineIntegration:

    def test_pipeline_with_registered_strategies(self, strategy_config):
        """
        Integration test: verifies that a full AnomalyDetectionEngine run with real strategies
        produces a valid, well-formed AnomalyResult attached to the prediction.
        Note: exact status depends on synthetic feature distances; we validate structure only.
        """
        reg = AnomalyRegistry()
        s_conf = ConfidenceThresholdStrategy(strategy_config)
        s_mah = MahalanobisOODStrategy(strategy_config)
        reg.register(s_conf)
        reg.register(s_mah)
        reg.initialize_all({})

        eng = AnomalyDetectionEngine(registry=reg)

        fake_pred = type("FakeResult", (), {
            "request_id": "req-e2e-1",
            "session_id": "sess-e2e-1",
            "modality": "chest_xray",
            "predicted_disease": "pneumonia",
            "confidence": 0.92,
            "extra": {},
        })()

        enriched = eng.analyze(fake_pred)
        anomaly = enriched.extra.get("anomaly_result")

        # Always validate the result structure and completeness
        assert anomaly is not None
        assert "status" in anomaly
        assert anomaly["status"] in (
            "KNOWN_DISEASE", "UNKNOWN_DISEASE", "OUT_OF_DISTRIBUTION",
            "LOW_CONFIDENCE", "POOR_QUALITY", "NOT_ANALYZED"
        )
        assert "recommended_action" in anomaly
        assert "risk_level" in anomaly
        assert "confidence_score" in anomaly
        assert anomaly["confidence_score"] == 0.92

    def test_concurrent_multi_modality_strategy_execution(self, strategy_config):
        reg = AnomalyRegistry()
        s_conf = ConfidenceThresholdStrategy(strategy_config)
        s_mah = MahalanobisOODStrategy(strategy_config)
        reg.register(s_conf)
        reg.register(s_mah)
        reg.initialize_all({})

        eng = AnomalyDetectionEngine(registry=reg)
        modalities = ["brain_mri", "chest_xray", "brain_ct", "histopathology", "retina", "skin", "ultrasound"]
        errors = []

        def _task(idx: int):
            try:
                mod = modalities[idx % len(modalities)]
                pred = type("FakeResult", (), {
                    "request_id": f"req-conc-strat-{idx}",
                    "session_id": f"sess-{idx}",
                    "modality": mod,
                    "predicted_disease": "normal",
                    "confidence": 0.88,
                    "extra": {},
                })()
                eng.analyze(pred)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_task, i) for i in range(35)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0
