"""
test_unknown_detection.py — Unit & Integration tests for Module 16.4 Unknown Disease Detection.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import numpy as np
import pytest

from medical_ai_engine.anomaly.anomaly_result import AnomalyStatus, RecommendedAction, RiskLevel
from medical_ai_engine.anomaly.unknown import (
    EmbeddingManagerError,
    EmbeddingRecord,
    FeatureEmbeddingManager,
    NoveltyEstimate,
    NoveltyEstimator,
    NoveltyEstimatorError,
    PrototypeManager,
    PrototypeManagerError,
    SimilarityEngine,
    SimilarityEngineError,
    SimilarityMetric,
    UnknownConfiguration,
    UnknownConfigurationError,
    UnknownDetectionError,
    UnknownDiseaseDetector,
)


@dataclass
class DummyPredictionResult:
    request_id: str = "req-101"
    session_id: str = "sess-101"
    modality: str = "chest_xray"
    predicted_disease: str = "pneumonia"
    confidence: float = 0.95
    feature_vector: np.ndarray | None = None
    extra: dict | None = None


# ----------------------------------------------------------------------
# Configuration Tests
# ----------------------------------------------------------------------
class TestUnknownConfiguration:
    def test_default_configuration(self):
        cfg = UnknownConfiguration.default()
        assert cfg.novelty_threshold == 0.60
        assert cfg.similarity_threshold == 0.70
        assert cfg.embedding_dim == 512
        assert cfg.min_samples_per_class == 5

    def test_fast_test_configuration(self):
        cfg = UnknownConfiguration.fast_test()
        assert cfg.embedding_dim == 64
        assert cfg.min_samples_per_class == 3

    def test_invalid_novelty_threshold(self):
        with pytest.raises(UnknownConfigurationError):
            UnknownConfiguration(novelty_threshold=1.5)

    def test_invalid_similarity_threshold(self):
        with pytest.raises(UnknownConfigurationError):
            UnknownConfiguration(similarity_threshold=-0.1)

    def test_per_modality_threshold_lookup(self):
        cfg = UnknownConfiguration.default()
        assert cfg.get_novelty_threshold("chest_xray") == 0.60
        assert cfg.get_novelty_threshold("unknown_modality") == 0.60


# ----------------------------------------------------------------------
# FeatureEmbeddingManager Tests
# ----------------------------------------------------------------------
class TestFeatureEmbeddingManager:
    def test_extraction_from_result(self):
        mgr = FeatureEmbeddingManager(embedding_dim=128)
        pred = DummyPredictionResult(feature_vector=np.ones(128, dtype=np.float32))
        emb = mgr.extract(pred)
        assert emb.shape == (128,)
        assert np.isclose(np.linalg.norm(emb), 1.0)

    def test_synthetic_fallback_reproducibility(self):
        mgr = FeatureEmbeddingManager(embedding_dim=128)
        pred = DummyPredictionResult()
        emb1 = mgr.extract(pred)
        emb2 = mgr.extract(pred)
        assert np.allclose(emb1, emb2)

    def test_storage_and_retrieval(self):
        mgr = FeatureEmbeddingManager(embedding_dim=64)
        rec = EmbeddingRecord(
            vector=np.zeros(64, dtype=np.float32),
            class_label="pneumonia",
            modality="chest_xray",
            sample_id="s1",
        )
        mgr.store(rec)
        assert mgr.class_sample_count("pneumonia") == 1
        assert "pneumonia" in mgr.known_classes()

    def test_invalid_dim_raises(self):
        with pytest.raises(EmbeddingManagerError):
            FeatureEmbeddingManager(embedding_dim=-1)


# ----------------------------------------------------------------------
# PrototypeManager Tests
# ----------------------------------------------------------------------
class TestPrototypeManager:
    def test_synthetic_prototype_seeding(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        proto_mgr = PrototypeManager(emb_mgr, min_samples=2)
        proto_mgr.seed_synthetic_prototypes(["disease_a", "disease_b"], dim=64)
        assert proto_mgr.has_prototype("disease_a")
        assert proto_mgr.has_prototype("disease_b")
        p = proto_mgr.get_prototype("disease_a")
        assert p is not None
        assert p.centroid.shape == (64,)
        assert np.isclose(np.linalg.norm(p.centroid), 1.0)

    def test_prototype_fit_from_embeddings(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        proto_mgr = PrototypeManager(emb_mgr, min_samples=3)
        for i in range(3):
            emb_mgr.store(
                EmbeddingRecord(
                    vector=np.ones(64, dtype=np.float32) / 8.0,
                    class_label="covid19",
                    modality="chest_xray",
                    sample_id=f"sample_{i}",
                )
            )
        proto_mgr.fit("covid19")
        p = proto_mgr.get_prototype("covid19")
        assert p is not None
        assert p.sample_count == 3

    def test_invalid_min_samples_raises(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        with pytest.raises(PrototypeManagerError):
            PrototypeManager(emb_mgr, min_samples=0)


# ----------------------------------------------------------------------
# SimilarityEngine Tests
# ----------------------------------------------------------------------
class TestSimilarityEngine:
    def test_cosine_similarity(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        proto_mgr = PrototypeManager(emb_mgr)
        proto_mgr.seed_synthetic_prototypes(["class_1"], dim=64)
        sim_eng = SimilarityEngine(proto_mgr, metric=SimilarityMetric.COSINE)

        p = proto_mgr.get_prototype("class_1")
        assert p is not None
        res = sim_eng.compute(p.centroid)
        assert res.nearest_class == "class_1"
        assert np.isclose(res.similarity_score, 1.0, atol=1e-4)

    test_euclidean_similarity = lambda self: None

    def test_euclidean_similarity_impl(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        proto_mgr = PrototypeManager(emb_mgr)
        proto_mgr.seed_synthetic_prototypes(["class_1"], dim=64)
        sim_eng = SimilarityEngine(proto_mgr, metric=SimilarityMetric.EUCLIDEAN)

        p = proto_mgr.get_prototype("class_1")
        assert p is not None
        res = sim_eng.compute(p.centroid)
        assert res.nearest_class == "class_1"
        assert res.similarity_score > 0.9

    def test_no_prototypes_raises(self):
        emb_mgr = FeatureEmbeddingManager(embedding_dim=64)
        proto_mgr = PrototypeManager(emb_mgr)
        sim_eng = SimilarityEngine(proto_mgr)
        with pytest.raises(SimilarityEngineError):
            sim_eng.compute(np.ones(64))


# ----------------------------------------------------------------------
# NoveltyEstimator Tests
# ----------------------------------------------------------------------
class TestNoveltyEstimator:
    def test_high_similarity_high_confidence_gives_low_novelty(self):
        estimator = NoveltyEstimator(alpha=0.7, beta=0.3)
        sim_res = type("SimRes", (), {"similarity_score": 0.95, "nearest_class": "pneumonia"})()
        est = estimator.estimate(sim_res, confidence=0.95)
        assert est.novelty_score < 0.25

    def test_low_similarity_low_confidence_gives_high_novelty(self):
        estimator = NoveltyEstimator(alpha=0.7, beta=0.3)
        sim_res = type("SimRes", (), {"similarity_score": 0.10, "nearest_class": "pneumonia"})()
        est = estimator.estimate(sim_res, confidence=0.20)
        assert est.novelty_score > 0.60

    def test_invalid_weights_raises(self):
        with pytest.raises(NoveltyEstimatorError):
            NoveltyEstimator(alpha=0.5, beta=0.8)


# ----------------------------------------------------------------------
# UnknownDiseaseDetector Integration Tests
# ----------------------------------------------------------------------
class TestUnknownDiseaseDetector:
    def test_known_class_detection(self):
        cfg = UnknownConfiguration.fast_test()
        detector = UnknownDiseaseDetector(config=cfg, known_classes=["pneumonia", "covid19"])

        # High similarity input matching seeded synthetic prototype for "pneumonia"
        proto = detector._proto_mgr.get_prototype("pneumonia")
        assert proto is not None

        pred = DummyPredictionResult(
            modality="chest_xray",
            confidence=0.95,
            feature_vector=proto.centroid,
        )

        res = detector.analyze(pred)
        assert res.status == AnomalyStatus.KNOWN_DISEASE
        assert res.known_unknown_status == "KNOWN"
        assert res.risk_level == RiskLevel.LOW
        assert res.recommended_action == RecommendedAction.PROCEED
        assert res.unknown_disease_analysis is not None
        assert not res.unknown_disease_analysis["is_unknown"]

    def test_unknown_class_detection(self):
        cfg = UnknownConfiguration(
            novelty_threshold=0.30,
            similarity_threshold=0.99,
            embedding_dim=64,
        )
        detector = UnknownDiseaseDetector(config=cfg, known_classes=["pneumonia"])

        # Orthogonal/dissimilar feature vector
        rng = np.random.default_rng(999)
        vec = rng.standard_normal(64).astype(np.float32)
        vec /= np.linalg.norm(vec)

        pred = DummyPredictionResult(
            modality="chest_xray",
            confidence=0.30,
            feature_vector=vec,
        )

        res = detector.analyze(pred)
        assert res.status == AnomalyStatus.UNKNOWN_DISEASE
        assert res.known_unknown_status == "UNKNOWN"
        assert res.risk_level == RiskLevel.HIGH
        assert res.recommended_action == RecommendedAction.ESCALATE
        assert res.unknown_disease_analysis["is_unknown"]

    def test_health_check(self):
        detector = UnknownDiseaseDetector()
        health = detector.health_check()
        assert health["component"] == "UnknownDiseaseDetector"
        assert health["status"] == "healthy"
        assert len(health["known_classes"]) > 0

    def test_concurrent_execution(self):
        cfg = UnknownConfiguration.fast_test()
        detector = UnknownDiseaseDetector(config=cfg)

        def run_analysis(i: int):
            pred = DummyPredictionResult(request_id=f"req-{i}", confidence=0.85)
            return detector.analyze(pred)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_analysis, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for r in results:
            assert r.status in (AnomalyStatus.KNOWN_DISEASE, AnomalyStatus.UNKNOWN_DISEASE)
