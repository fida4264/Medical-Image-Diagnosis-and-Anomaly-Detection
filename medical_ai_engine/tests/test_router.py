"""
Medical AI Engine — Modality Router Test Suite (Module 14.2.3)

Tests:
  - Routing decision outcomes (ROUTE_TO_MODEL, LOW_CONFIDENCE, UNSUPPORTED_MODALITY, MODEL_NOT_REGISTERED)
  - Per-modality confidence thresholds
  - Fallback policy evaluations
  - Manual modality override
  - Registry integration & adapter lookup
  - Context population (ctx.modality_prediction, metrics, logs, warnings, errors)
  - Thread safety & concurrent routing execution
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
import pytest
from PIL import Image

from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.routing.confidence_policy import ConfidencePolicy
from medical_ai_engine.routing.fallback_policy import FallbackPolicy
from medical_ai_engine.routing.modality_router import ModalityRouter
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import (
    RoutingDecision,
    RoutingOutcome,
)
from medical_ai_engine.routing.routing_policy import RoutingPolicy
from modality_classifier.inference.predictor import ModalityPredictionResult


# ── Mock Classes for Isolated Unit Testing ─────────────────────────────────────

class MockModalityPredictor:
    """Mock Module 6 predictor returning controllable outputs."""

    def __init__(self, default_modality: str = "skin", default_confidence: float = 0.95):
        self.default_modality = default_modality
        self.default_confidence = default_confidence

    def predict(self, image_input: Any, top_k: int = 3) -> ModalityPredictionResult:
        return ModalityPredictionResult(
            predicted_modality=self.default_modality,
            confidence=self.default_confidence,
            top_k_predictions=[
                {"modality": self.default_modality, "confidence": self.default_confidence},
                {"modality": "chest_xray", "confidence": 0.03},
                {"modality": "brain_mri", "confidence": 0.02},
            ],
            inference_time_ms=12.5,
            routing_destination=f"Module ({self.default_modality})",
        )


class MockAdapter(BasePredictorAdapter):
    """Mock IPredictorAdapter for registry test setup."""
    _MODALITY = "skin"

    @property
    def modality(self) -> str:
        return self._MODALITY

    def load_model(self, checkpoint_path=None, label_map_path=None):
        self._is_loaded = True
        self._predictor = object()

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        return {"predicted_class": "mock", "confidence": 0.99}


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


def build_router(
    pred_modality: str = "skin",
    pred_confidence: float = 0.95,
    register_adapters: bool = True,
    custom_config: Optional[RoutingConfiguration] = None,
) -> ModalityRouter:
    cfg = custom_config or RoutingConfiguration()
    reg_cfg = RegistryConfiguration(auto_discover_adapters=False)
    registry = ModelRegistry(config=reg_cfg)

    if register_adapters:
        for mod in cfg.supported_modalities:
            class SubAdapter(MockAdapter):
                _MODALITY = mod
            SubAdapter.__name__ = f"Mock{mod.title().replace('_','')}Adapter"
            registry.register_adapter(mod, SubAdapter)

    mock_pred = MockModalityPredictor(
        default_modality=pred_modality,
        default_confidence=pred_confidence,
    )

    router = ModalityRouter(
        config=cfg,
        model_registry=registry,
        predictor=mock_pred,
    )
    return router


def make_ctx(
    override: Optional[str] = None,
    req_id: str = "req_test_123",
) -> EngineContext:
    ctx = EngineContext(
        request_id=req_id,
        session_id="sess_test_123",
        manual_modality_override=override,
        image=Image.new("RGB", (224, 224), color=(255, 0, 0)),
    )
    return ctx


# ── Tests: Routing Decision Outcomes ──────────────────────────────────────────

def test_successful_route_to_model():
    router = build_router(pred_modality="skin", pred_confidence=0.95)
    ctx = make_ctx()

    decision = router.route(ctx)

    assert decision.outcome == RoutingOutcome.ROUTE_TO_MODEL
    assert decision.is_success is True
    assert decision.detected_modality == "skin"
    assert decision.adapter_key == "skin"
    assert decision.confidence == 0.95

    # Context populated
    assert ctx.modality_prediction is not None
    assert ctx.modality_prediction["detected_modality"] == "skin"
    assert ctx.modality_prediction["adapter_key"] == "skin"
    assert "routing_ms" in ctx.performance_metrics


def test_low_confidence_rejection():
    # skin threshold is 0.85, set confidence to 0.75
    router = build_router(pred_modality="skin", pred_confidence=0.75)
    ctx = make_ctx()

    decision = router.route(ctx)

    assert decision.outcome == RoutingOutcome.LOW_CONFIDENCE
    assert decision.is_success is False
    assert decision.is_low_confidence is True
    assert decision.adapter_key is None
    assert "below required threshold" in decision.reason

    # Errors populated in context
    assert len(ctx.errors) == 1
    assert ctx.errors[0]["stage"] == "ModalityRouter"


def test_unsupported_modality_rejection():
    # 'dental_xray' is not in supported_modalities
    router = build_router(pred_modality="dental_xray", pred_confidence=0.99)
    ctx = make_ctx()

    decision = router.route(ctx)

    assert decision.outcome == RoutingOutcome.UNSUPPORTED_MODALITY
    assert decision.is_success is False
    assert decision.is_unsupported is True
    assert "not in the supported modality set" in decision.reason


def test_model_not_registered_rejection():
    # Skin adapter is NOT registered in ModelRegistry
    router = build_router(pred_modality="skin", pred_confidence=0.95, register_adapters=False)
    ctx = make_ctx()

    decision = router.route(ctx)

    assert decision.outcome == RoutingOutcome.MODEL_NOT_REGISTERED
    assert decision.is_success is False
    assert decision.is_not_registered is True
    assert "No adapter is registered" in decision.reason


def test_manual_modality_override():
    router = build_router(pred_modality="chest_xray", pred_confidence=0.99)
    ctx = make_ctx(override="brain_mri")

    decision = router.route(ctx)

    assert decision.outcome == RoutingOutcome.ROUTE_TO_MODEL
    assert decision.detected_modality == "brain_mri"
    assert decision.adapter_key == "brain_mri"
    assert decision.routing_mode == "manual_override"
    assert decision.confidence == 1.0


# ── Tests: Per-Modality Thresholds ────────────────────────────────────────────

def test_per_modality_threshold_evaluation():
    cfg = RoutingConfiguration()

    # brain_mri threshold = 0.90
    router_mri_pass = build_router(pred_modality="brain_mri", pred_confidence=0.91, custom_config=cfg)
    assert router_mri_pass.route(make_ctx()).outcome == RoutingOutcome.ROUTE_TO_MODEL

    router_mri_fail = build_router(pred_modality="brain_mri", pred_confidence=0.89, custom_config=cfg)
    assert router_mri_fail.route(make_ctx()).outcome == RoutingOutcome.LOW_CONFIDENCE

    # ultrasound threshold = 0.83
    router_us_pass = build_router(pred_modality="ultrasound", pred_confidence=0.84, custom_config=cfg)
    assert router_us_pass.route(make_ctx()).outcome == RoutingOutcome.ROUTE_TO_MODEL

    router_us_fail = build_router(pred_modality="ultrasound", pred_confidence=0.80, custom_config=cfg)
    assert router_us_fail.route(make_ctx()).outcome == RoutingOutcome.LOW_CONFIDENCE


def test_global_confidence_floor():
    cfg = RoutingConfiguration(global_confidence_floor=0.60)
    # Confidence 0.50 is below global floor 0.60
    router = build_router(pred_modality="ultrasound", pred_confidence=0.50, custom_config=cfg)
    decision = router.route(make_ctx())
    assert decision.outcome == RoutingOutcome.LOW_CONFIDENCE


# ── Tests: Policies ────────────────────────────────────────────────────────────

def test_confidence_policy_standalone():
    cfg = RoutingConfiguration()
    cp = ConfidencePolicy(config=cfg)

    # Below threshold -> decision returned
    res = cp.evaluate(modality="brain_mri", confidence=0.80, request_id="r1")
    assert res is not None
    assert res.outcome == RoutingOutcome.LOW_CONFIDENCE

    # Above threshold -> None returned
    res_pass = cp.evaluate(modality="brain_mri", confidence=0.95, request_id="r1")
    assert res_pass is None


def test_fallback_policy_standalone():
    cfg = RoutingConfiguration()
    fp = FallbackPolicy(config=cfg)

    # Modality check
    res_unsupported = fp.evaluate_modality(modality="pet_scan", confidence=0.95, request_id="r1")
    assert res_unsupported is not None
    assert res_unsupported.outcome == RoutingOutcome.UNSUPPORTED_MODALITY

    res_supported = fp.evaluate_modality(modality="brain_mri", confidence=0.95, request_id="r1")
    assert res_supported is None

    # Adapter check
    res_no_adapter = fp.evaluate_adapter(modality="brain_mri", adapter_registered=False, request_id="r1")
    assert res_no_adapter is not None
    assert res_no_adapter.outcome == RoutingOutcome.MODEL_NOT_REGISTERED


def test_routing_policy_composite():
    cfg = RoutingConfiguration()
    rp = RoutingPolicy(config=cfg)

    # Test order of evaluation: unsupported check triggers first
    res = rp.apply(
        modality="pet_scan",
        confidence=0.10,  # Also low confidence, but unsupported comes first
        adapter_registered=False,
        request_id="r1",
    )
    assert res is not None
    assert res.outcome == RoutingOutcome.UNSUPPORTED_MODALITY


# ── Tests: Thread Safety & Concurrent Routing ─────────────────────────────────

def test_concurrent_routing_execution():
    router = build_router(pred_modality="skin", pred_confidence=0.95)
    decisions = []
    errors = []

    def worker(idx: int):
        try:
            ctx = make_ctx(req_id=f"req_thread_{idx}")
            dec = router.route(ctx)
            decisions.append(dec)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(decisions) == 20
    assert all(d.outcome == RoutingOutcome.ROUTE_TO_MODEL for d in decisions)


# ── Tests: Health Check ───────────────────────────────────────────────────────

def test_router_health_check():
    router = build_router()
    health = router.health_check()

    assert health["status"] in ["HEALTHY", "DEGRADED"]
    assert health["router_name"] == "UnifiedModalityRouter"
    assert health["registry_available"] is True
    assert len(health["supported_modalities"]) == 7
