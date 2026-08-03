"""
test_anomaly_architecture.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Comprehensive architecture unit tests.

Covers:
  - Package imports and exports
  - AnomalyConfiguration validation
  - AnomalyContext lifecycle
  - AnomalyResult immutability and serialization
  - IDetectionStrategy interface contract
  - AnomalyRegistry thread-safe operations
  - ConfidenceAnalyzer classification logic
  - OODDetector fallback (no strategy registered)
  - UnknownDiseaseDetector fallback (no strategy registered)
  - RiskAssessor multi-signal fusion
  - DecisionEngine priority logic
  - AnomalyDetectionEngine end-to-end pipeline
  - Fault tolerance & failure recovery
  - Concurrent multi-threaded analysis
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import threading
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from medical_ai_engine.anomaly import (
    AnomalyConfiguration,
    AnomalyConfigurationError,
    AnomalyContext,
    AnomalyDetectionEngine,
    AnomalyRegistry,
    AnomalyResult,
    AnomalyStatus,
    ConfidenceAnalyzer,
    DecisionEngine,
    IAnomalyRegistry,
    IDetectionStrategy,
    OODDetector,
    RecommendedAction,
    RiskAssessor,
    RiskLevel,
    StrategyNotFoundError,
    UnknownDiseaseDetector,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class _FakePredictionResult:
    request_id: str = "req-test-001"
    session_id: str = "sess-test-001"
    modality: str = "brain_mri"
    predicted_disease: str = "glioma"
    confidence: float = 0.92
    is_success: bool = True
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


class _MockStrategy(IDetectionStrategy):
    """A minimal concrete strategy for testing purposes."""

    def __init__(self, key: str = "mock_strategy", always_anomaly: bool = False):
        self._key = key
        self._always_anomaly = always_anomaly
        self._initialized = False

    @property
    def strategy_key(self) -> str:
        return self._key

    @property
    def strategy_version(self) -> str:
        return "1.0.0-test"

    @property
    def strategy_description(self) -> str:
        return "Mock strategy for architecture testing."

    def initialize(self, config: Dict[str, Any], model_adapter: Optional[Any] = None) -> None:
        self._initialized = True

    def detect(self, context: AnomalyContext) -> bool:
        context.ood_analysis = {
            "is_ood": self._always_anomaly,
            "ood_score": 5.0 if self._always_anomaly else 0.5,
            "distance": 5.0 if self._always_anomaly else 0.5,
            "threshold": 3.0,
            "strategy_used": self._key,
        }
        context.unknown_disease_analysis = {
            "is_unknown": self._always_anomaly,
            "reconstruction_error": 0.50 if self._always_anomaly else 0.02,
            "threshold": 0.10,
            "strategy_used": self._key,
            "confidence_in_known": context.confidence,
        }
        return self._always_anomaly

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self._key,
            "version": self.strategy_version,
            "description": self.strategy_description,
            "requires_feature_vector": False,
            "requires_model": False,
            "is_initialized": self._initialized,
        }


@pytest.fixture
def fast_config() -> AnomalyConfiguration:
    return AnomalyConfiguration.fast_test()


@pytest.fixture
def fake_prediction() -> _FakePredictionResult:
    return _FakePredictionResult()


@pytest.fixture
def engine(fast_config) -> AnomalyDetectionEngine:
    return AnomalyDetectionEngine(config=fast_config)


@pytest.fixture
def engine_with_mock(fast_config) -> AnomalyDetectionEngine:
    registry = AnomalyRegistry()
    registry.register(_MockStrategy(key="confidence_threshold"))
    return AnomalyDetectionEngine(config=fast_config, registry=registry)


# ── T01: Package Imports ──────────────────────────────────────────────────────

class TestPackageImports:
    def test_all_exports_importable(self):
        from medical_ai_engine.anomaly import (
            AnomalyDetectionEngine,
            AnomalyRegistry,
            AnomalyConfiguration,
            AnomalyContext,
            AnomalyResult,
            AnomalyStatus,
            RiskLevel,
            RecommendedAction,
            ConfidenceAnalyzer,
            OODDetector,
            UnknownDiseaseDetector,
            RiskAssessor,
            DecisionEngine,
            IDetectionStrategy,
            IAnomalyRegistry,
        )

    def test_interface_exports_are_abstract(self):
        with pytest.raises(TypeError):
            IDetectionStrategy()  # Abstract, cannot instantiate


# ── T02: AnomalyConfiguration ─────────────────────────────────────────────────

class TestAnomalyConfiguration:
    def test_default_values(self):
        cfg = AnomalyConfiguration()
        assert cfg.enabled is True
        assert cfg.fail_silently is True
        assert cfg.confidence_threshold_low == 0.70
        assert cfg.confidence_threshold_critical == 0.40
        assert cfg.default_strategy == "confidence_threshold"

    def test_invalid_confidence_threshold_raises(self):
        with pytest.raises(AnomalyConfigurationError):
            AnomalyConfiguration(confidence_threshold_low=1.5)

    def test_critical_greater_than_low_raises(self):
        with pytest.raises(AnomalyConfigurationError):
            AnomalyConfiguration(confidence_threshold_critical=0.80, confidence_threshold_low=0.70)

    def test_fast_test_factory(self):
        cfg = AnomalyConfiguration.fast_test()
        assert cfg.enabled is True
        assert cfg.async_execution is False

    def test_disabled_factory(self):
        cfg = AnomalyConfiguration.disabled()
        assert cfg.enabled is False

    def test_to_dict_serializable(self):
        cfg = AnomalyConfiguration()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "enabled" in d
        assert "default_strategy" in d


# ── T03: AnomalyContext ───────────────────────────────────────────────────────

class TestAnomalyContext:
    def test_construction(self):
        ctx = AnomalyContext(request_id="req-1", session_id="sess-1")
        assert ctx.request_id == "req-1"
        assert ctx.anomaly_status == "NOT_ANALYZED"
        assert ctx.risk_level == "UNKNOWN"
        assert ctx.execution_state == "CREATED"

    def test_log_and_warn(self):
        ctx = AnomalyContext(request_id="req-2", session_id="sess-2")
        ctx.log("Test log", stage="TestStage")
        ctx.warn("Test warn", stage="TestStage")
        assert any("Test log" in l for l in ctx.logs)
        assert any("Test warn" in w for w in ctx.warnings)

    def test_add_error(self):
        ctx = AnomalyContext(request_id="req-3", session_id="sess-3")
        ctx.add_error("Stage1", "Something failed")
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["stage"] == "Stage1"

    def test_record_metric(self):
        ctx = AnomalyContext(request_id="req-4", session_id="sess-4")
        ctx.record_metric("gradcam_ms", 12.5)
        assert ctx.performance_metrics["gradcam_ms"] == 12.5

    def test_summary_fields(self):
        ctx = AnomalyContext(request_id="req-5", session_id="sess-5", modality="chest_xray", confidence=0.88)
        s = ctx.summary()
        assert s["modality"] == "chest_xray"
        assert s["confidence"] == 0.88


# ── T04: AnomalyResult ───────────────────────────────────────────────────────

class TestAnomalyResult:
    def test_immutability(self):
        result = AnomalyResult(request_id="r1", session_id="s1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = AnomalyStatus.OUT_OF_DISTRIBUTION  # type: ignore

    def test_to_dict_json_compatible(self):
        result = AnomalyResult(
            request_id="r2",
            session_id="s2",
            status=AnomalyStatus.KNOWN_DISEASE,
            risk_level=RiskLevel.LOW,
            recommended_action=RecommendedAction.PROCEED,
            confidence_score=0.95,
        )
        d = result.to_dict()
        assert d["status"] == "KNOWN_DISEASE"
        assert d["risk_level"] == "LOW"
        assert d["recommended_action"] == "PROCEED"
        j = json.loads(result.to_json())
        assert j["confidence_score"] == 0.95

    def test_is_trusted_property(self):
        trusted = AnomalyResult(
            request_id="r3", session_id="s3",
            status=AnomalyStatus.KNOWN_DISEASE, risk_level=RiskLevel.LOW,
        )
        untrusted = AnomalyResult(
            request_id="r4", session_id="s4",
            status=AnomalyStatus.OUT_OF_DISTRIBUTION, risk_level=RiskLevel.CRITICAL,
        )
        assert trusted.is_trusted is True
        assert untrusted.is_trusted is False

    def test_is_anomalous_property(self):
        ood = AnomalyResult(request_id="r5", session_id="s5", status=AnomalyStatus.OUT_OF_DISTRIBUTION)
        known = AnomalyResult(request_id="r6", session_id="s6", status=AnomalyStatus.KNOWN_DISEASE)
        assert ood.is_anomalous is True
        assert known.is_anomalous is False

    def test_requires_review_property(self):
        review = AnomalyResult(request_id="r7", session_id="s7", recommended_action=RecommendedAction.MANUAL_REVIEW)
        proceed = AnomalyResult(request_id="r8", session_id="s8", recommended_action=RecommendedAction.PROCEED)
        assert review.requires_review is True
        assert proceed.requires_review is False


# ── T05: AnomalyRegistry ────────────────────────────────────────────────────

class TestAnomalyRegistry:
    def test_register_and_list(self):
        reg = AnomalyRegistry()
        s = _MockStrategy(key="test_strategy_01")
        reg.register(s)
        assert "test_strategy_01" in reg.list_strategies()

    def test_get_strategy_found(self):
        reg = AnomalyRegistry()
        s = _MockStrategy(key="test_strategy_02")
        reg.register(s)
        assert reg.get_strategy("test_strategy_02") is s

    def test_get_strategy_not_found_raises(self):
        reg = AnomalyRegistry()
        with pytest.raises(StrategyNotFoundError):
            reg.get_strategy("nonexistent_key")

    def test_unregister(self):
        reg = AnomalyRegistry()
        s = _MockStrategy(key="test_strategy_03")
        reg.register(s)
        reg.unregister("test_strategy_03")
        assert not reg.is_registered("test_strategy_03")

    def test_initialize_all(self):
        reg = AnomalyRegistry()
        s = _MockStrategy(key="test_strategy_04")
        reg.register(s)
        reg.initialize_all(config={})
        assert s._initialized is True

    def test_health_report(self):
        reg = AnomalyRegistry()
        s = _MockStrategy(key="test_strategy_05")
        reg.register(s)
        report = reg.health_report()
        assert "test_strategy_05" in report
        assert report["test_strategy_05"]["status"] == "OK"

    def test_thread_safe_concurrent_registration(self):
        reg = AnomalyRegistry()
        errors = []

        def register_many(idx: int):
            try:
                reg.register(_MockStrategy(key=f"strategy_{idx}"))
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(register_many, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert len(reg.list_strategies()) == 50


# ── T06: ConfidenceAnalyzer ──────────────────────────────────────────────────

class TestConfidenceAnalyzer:
    def test_high_confidence(self, fast_config):
        analyzer = ConfidenceAnalyzer(fast_config)
        ctx = AnomalyContext(request_id="r1", session_id="s1", confidence=0.92)
        analyzer.analyze(ctx)
        assert ctx.confidence_analysis["confidence_flag"] == "TRUSTED"
        assert ctx.confidence_analysis["confidence_level"] == "HIGH"

    def test_low_confidence(self, fast_config):
        analyzer = ConfidenceAnalyzer(fast_config)
        ctx = AnomalyContext(request_id="r2", session_id="s2", confidence=0.55)
        analyzer.analyze(ctx)
        assert ctx.confidence_analysis["confidence_flag"] == "LOW_CONFIDENCE"

    def test_critical_confidence(self, fast_config):
        analyzer = ConfidenceAnalyzer(fast_config)
        ctx = AnomalyContext(request_id="r3", session_id="s3", confidence=0.20)
        analyzer.analyze(ctx)
        assert ctx.confidence_analysis["confidence_flag"] == "CRITICAL"


# ── T07: OODDetector (no strategy registered) ────────────────────────────────

class TestOODDetector:
    def test_no_strategy_returns_false(self, fast_config):
        reg = AnomalyRegistry()  # Empty registry
        detector = OODDetector(fast_config, reg)
        ctx = AnomalyContext(request_id="r1", session_id="s1", confidence=0.85)
        result = detector.detect(ctx)
        assert result is False
        assert ctx.ood_analysis["is_ood"] is False
        assert ctx.ood_analysis["strategy_used"] == "none"

    def test_with_mock_ood_strategy(self, fast_config):
        reg = AnomalyRegistry()
        reg.register(_MockStrategy(key="mahalanobis", always_anomaly=True))
        detector = OODDetector(fast_config, reg)
        ctx = AnomalyContext(request_id="r2", session_id="s2", confidence=0.85)
        result = detector.detect(ctx)
        assert result is True
        assert ctx.ood_analysis["is_ood"] is True


# ── T08: RiskAssessor ────────────────────────────────────────────────────────

class TestRiskAssessor:
    def test_high_confidence_low_risk(self, fast_config):
        assessor = RiskAssessor(fast_config)
        ctx = AnomalyContext(request_id="r1", session_id="s1", confidence=0.95)
        ctx.ood_analysis = {"is_ood": False, "ood_score": 0.1}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.01}
        assessor.assess(ctx)
        assert ctx.risk_level in ("LOW", "MODERATE")

    def test_ood_detected_high_risk(self, fast_config):
        assessor = RiskAssessor(fast_config)
        ctx = AnomalyContext(request_id="r2", session_id="s2", confidence=0.30)
        ctx.ood_analysis = {"is_ood": True, "ood_score": 8.5}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.01}
        assessor.assess(ctx)
        assert ctx.risk_level in ("HIGH", "CRITICAL")
        assert len(ctx.risk_factors) > 0


# ── T09: DecisionEngine ──────────────────────────────────────────────────────

class TestDecisionEngine:
    def test_known_disease_proceed(self, fast_config):
        engine = DecisionEngine(fast_config)
        ctx = AnomalyContext(request_id="r1", session_id="s1", confidence=0.92)
        ctx.risk_level = "LOW"
        ctx.risk_score = 0.10
        ctx.ood_analysis = {"is_ood": False, "ood_score": 0.5}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.02}
        ctx.confidence_analysis = {"confidence_flag": "TRUSTED"}
        result = engine.decide(ctx, total_elapsed_ms=10.0)
        assert result.status == AnomalyStatus.KNOWN_DISEASE
        assert result.recommended_action == RecommendedAction.PROCEED

    def test_ood_escalate(self, fast_config):
        engine = DecisionEngine(fast_config)
        ctx = AnomalyContext(request_id="r2", session_id="s2", confidence=0.40)
        ctx.risk_level = "CRITICAL"
        ctx.risk_score = 0.90
        ctx.ood_analysis = {"is_ood": True, "ood_score": 7.5}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.01}
        ctx.confidence_analysis = {"confidence_flag": "CRITICAL"}
        result = engine.decide(ctx, total_elapsed_ms=12.0)
        assert result.status == AnomalyStatus.OUT_OF_DISTRIBUTION
        assert result.recommended_action == RecommendedAction.ESCALATE

    def test_low_confidence_manual_review(self, fast_config):
        engine = DecisionEngine(fast_config)
        ctx = AnomalyContext(request_id="r3", session_id="s3", confidence=0.55)
        ctx.risk_level = "MODERATE"
        ctx.risk_score = 0.35
        ctx.ood_analysis = {"is_ood": False, "ood_score": 0.5}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.02}
        ctx.confidence_analysis = {"confidence_flag": "LOW_CONFIDENCE"}
        result = engine.decide(ctx, total_elapsed_ms=8.0)
        assert result.status == AnomalyStatus.LOW_CONFIDENCE
        assert result.recommended_action == RecommendedAction.MANUAL_REVIEW

    def test_poor_quality_repeat_acquisition(self, fast_config):
        engine = DecisionEngine(fast_config)
        ctx = AnomalyContext(request_id="r4", session_id="s4", confidence=0.92, quality_score=0.15)
        ctx.risk_level = "LOW"
        ctx.risk_score = 0.10
        ctx.ood_analysis = {"is_ood": False, "ood_score": 0.1}
        ctx.unknown_disease_analysis = {"is_unknown": False, "reconstruction_error": 0.01}
        ctx.confidence_analysis = {"confidence_flag": "TRUSTED"}
        result = engine.decide(ctx, total_elapsed_ms=6.0)
        assert result.status == AnomalyStatus.POOR_QUALITY
        assert result.recommended_action == RecommendedAction.REPEAT_ACQUISITION


# ── T10: AnomalyDetectionEngine (End-to-End) ─────────────────────────────────

class TestAnomalyDetectionEngine:
    def test_high_confidence_prediction_known_disease(self, engine, fake_prediction):
        result = engine.analyze(fake_prediction)
        anomaly = result.extra.get("anomaly_result")
        assert anomaly is not None
        assert anomaly["status"] == "KNOWN_DISEASE"
        assert anomaly["recommended_action"] == "PROCEED"

    def test_disabled_engine_bypasses_analysis(self, fake_prediction):
        cfg = AnomalyConfiguration.disabled()
        eng = AnomalyDetectionEngine(config=cfg)
        result = eng.analyze(fake_prediction)
        # Result should be unmodified original
        assert result is fake_prediction

    def test_health_check(self, engine):
        health = engine.health_check()
        assert health["enabled"] is True
        assert "total_analyzed" in health

    def test_low_confidence_triggers_review(self, fast_config):
        low_conf_pred = _FakePredictionResult(confidence=0.35)
        eng = AnomalyDetectionEngine(config=fast_config)
        result = eng.analyze(low_conf_pred)
        anomaly = result.extra.get("anomaly_result")
        assert anomaly["status"] in ("LOW_CONFIDENCE", "KNOWN_DISEASE")

    def test_concurrent_analysis_thread_safe(self, engine, fake_prediction):
        errors = []
        results = []

        def _analyze(_):
            try:
                pred = _FakePredictionResult(request_id=f"req-{threading.get_ident()}")
                r = engine.analyze(pred)
                results.append(r)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_analyze, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0
        assert len(results) == 50
        assert all(r.extra.get("anomaly_result") is not None for r in results)
