"""
test_decision_fusion.py — Unit & Integration tests for Module 16.5 Risk Assessment & Decision Fusion.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import pytest

from medical_ai_engine.anomaly.anomaly_result import AnomalyStatus, RecommendedAction, RiskLevel
from medical_ai_engine.anomaly.decision import (
    DecisionConfiguration,
    DecisionConfigurationError,
    DecisionEngine,
    DecisionError,
    DefaultPriorityPolicy,
    FusionEngine,
    FusionEngineError,
    PolicyError,
    PolicyRegistry,
    PriorityManager,
    PriorityError,
    RiskAssessor,
    SignalType,
    WeightedFusionPolicy,
)


@dataclass
class DummyPrediction:
    request_id: str = "req-fuse-1"
    session_id: str = "sess-fuse-1"
    modality: str = "chest_xray"
    confidence: float = 0.92


# ----------------------------------------------------------------------
# Configuration Tests
# ----------------------------------------------------------------------
class TestDecisionConfiguration:
    def test_default_configuration(self):
        cfg = DecisionConfiguration.default()
        assert cfg.policy_name == "default_priority"
        assert cfg.high_risk_threshold == 0.70
        assert cfg.critical_risk_threshold == 0.88

    def test_invalid_thresholds_raise(self):
        with pytest.raises(DecisionConfigurationError):
            DecisionConfiguration(high_risk_threshold=0.90, critical_risk_threshold=0.85)

    def test_per_modality_override(self):
        cfg = DecisionConfiguration.default()
        assert cfg.per_modality_thresholds["chest_xray"] == 0.70


# ----------------------------------------------------------------------
# PriorityManager Tests
# ----------------------------------------------------------------------
class TestPriorityManager:
    def test_priority_hierarchy_resolution(self):
        pm = PriorityManager()
        # Unknown disease should beat OOD and Low Confidence
        signals = [SignalType.LOW_CONFIDENCE, SignalType.OOD, SignalType.UNKNOWN_DISEASE]
        highest = pm.resolve_highest_priority(signals)
        assert highest == SignalType.UNKNOWN_DISEASE

    def test_ood_beats_low_confidence(self):
        pm = PriorityManager()
        signals = [SignalType.LOW_CONFIDENCE, SignalType.OOD]
        highest = pm.resolve_highest_priority(signals)
        assert highest == SignalType.OOD

    def test_empty_signals_returns_known_disease(self):
        pm = PriorityManager()
        assert pm.resolve_highest_priority([]) == SignalType.KNOWN_DISEASE


# ----------------------------------------------------------------------
# FusionEngine & Policy Tests
# ----------------------------------------------------------------------
class TestFusionEngine:
    def test_default_priority_policy_unknown_disease(self):
        fusion = FusionEngine()
        out = fusion.fuse(
            unknown_signal={"is_unknown": True, "novelty_score": 0.82},
            ood_signal={"is_ood": True, "ood_score": 5.0},
            confidence=0.90,
        )
        assert out.status == AnomalyStatus.UNKNOWN_DISEASE
        assert out.known_unknown_status == "UNKNOWN"
        assert out.recommended_action == RecommendedAction.ESCALATE
        assert out.risk_level == RiskLevel.HIGH

    def test_weighted_fusion_policy(self):
        cfg = DecisionConfiguration.weighted()
        fusion = FusionEngine(config=cfg)
        out = fusion.fuse(
            unknown_signal={"novelty_score": 0.90},
            ood_signal={"ood_score": 8.0},
            confidence=0.20,
        )
        assert out.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert out.risk_score > 0.70

    def test_policy_switch(self):
        fusion = FusionEngine()
        assert fusion.current_policy_name == "default_priority"
        fusion.set_policy("weighted_fusion")
        assert fusion.current_policy_name == "weighted_fusion"

    def test_unknown_policy_raises(self):
        registry = PolicyRegistry()
        with pytest.raises(PolicyError):
            registry.get("non_existent_policy")


# ----------------------------------------------------------------------
# RiskAssessor Tests
# ----------------------------------------------------------------------
class TestRiskAssessor:
    def test_risk_level_assessment(self):
        assessor = RiskAssessor()
        out = assessor.assess(
            status=AnomalyStatus.UNKNOWN_DISEASE,
            risk_score=0.85,
            confidence=0.50,
            signals={},
            modality="chest_xray",
        )
        assert out.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(out.risk_factors) > 0

    def test_low_risk_assessment(self):
        assessor = RiskAssessor()
        out = assessor.assess(
            status=AnomalyStatus.KNOWN_DISEASE,
            risk_score=0.10,
            confidence=0.95,
            signals={},
        )
        assert out.risk_level == RiskLevel.LOW


# ----------------------------------------------------------------------
# DecisionEngine Master Integration Tests
# ----------------------------------------------------------------------
class TestDecisionEngineMaster:
    def test_evaluate_signals_known_disease(self):
        engine = DecisionEngine()
        pred = DummyPrediction(confidence=0.95)
        res = engine.evaluate_signals(pred)

        assert res.status == AnomalyStatus.KNOWN_DISEASE
        assert res.known_unknown_status == "KNOWN"
        assert res.risk_level == RiskLevel.LOW
        assert res.recommended_action == RecommendedAction.PROCEED
        assert res.risk_score < 0.20

    def test_evaluate_signals_conflicting_signals(self):
        """
        When both Unknown Disease and OOD are present, Priority Policy ensures
        Unknown Disease takes precedence.
        """
        engine = DecisionEngine()
        pred = DummyPrediction(confidence=0.30)
        res = engine.evaluate_signals(
            prediction_result=pred,
            confidence_signal={"is_low_confidence": True},
            ood_signal={"is_ood": True, "ood_score": 4.5},
            unknown_signal={"is_unknown": True, "novelty_score": 0.88},
        )

        assert res.status == AnomalyStatus.UNKNOWN_DISEASE
        assert res.known_unknown_status == "UNKNOWN"
        assert res.recommended_action == RecommendedAction.ESCALATE
        assert res.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_multimodality_support(self):
        engine = DecisionEngine()
        for modality in ["brain_mri", "chest_xray", "histopathology", "ultrasound"]:
            pred = DummyPrediction(modality=modality, confidence=0.88)
            res = engine.evaluate_signals(pred)
            assert res.request_id == "req-fuse-1"
            assert res.status == AnomalyStatus.KNOWN_DISEASE

    def test_thread_safety_concurrent_execution(self):
        engine = DecisionEngine()

        def run_eval(i: int):
            pred = DummyPrediction(request_id=f"req-{i}", confidence=0.90)
            return engine.evaluate_signals(
                pred,
                ood_signal={"is_ood": (i % 2 == 0), "ood_score": float(i)},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_eval, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for r in results:
            assert r.status in (AnomalyStatus.KNOWN_DISEASE, AnomalyStatus.OUT_OF_DISTRIBUTION)

    def test_health_check(self):
        engine = DecisionEngine()
        health = engine.health_check()
        assert health["component"] == "DecisionEngine"
        assert health["status"] == "healthy"
