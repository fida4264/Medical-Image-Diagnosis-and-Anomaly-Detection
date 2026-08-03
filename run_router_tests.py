"""
Medical AI Engine Router — Test Runner Script (Module 14.2.3)

Demonstrates:
  1. Routing decision outcomes (ROUTE_TO_MODEL, LOW_CONFIDENCE, UNSUPPORTED_MODALITY, MODEL_NOT_REGISTERED)
  2. Manual modality override
  3. Context mutation verification
  4. Router health check
  5. PyTest suite execution
"""

import sys
import pytest
from PIL import Image

from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.routing.modality_router import ModalityRouter
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import RoutingOutcome
from modality_classifier.inference.predictor import ModalityPredictionResult


class MockPredictor:
    def __init__(self, modality="skin", confidence=0.95):
        self.modality = modality
        self.confidence = confidence

    def predict(self, image_input, top_k=3):
        return ModalityPredictionResult(
            predicted_modality=self.modality,
            confidence=self.confidence,
            top_k_predictions=[{"modality": self.modality, "confidence": self.confidence}],
            inference_time_ms=10.0,
            routing_destination="Module (Demo)",
        )


class MockAdapter(BasePredictorAdapter):
    _MODALITY = "skin"
    @property
    def modality(self): return self._MODALITY
    def load_model(self, checkpoint_path=None, label_map_path=None): self._is_loaded = True
    def predict(self, image): return {"class": "demo"}


def main():
    print("=" * 66)
    print("      Module 14.2.3: Unified Modality Router Runner       ")
    print("=" * 66)

    ModelRegistry.reset_instance()
    reg_cfg = RegistryConfiguration(auto_discover_adapters=False)
    reg = ModelRegistry(config=reg_cfg)

    # Register adapters
    for mod in ["brain_mri", "chest_xray", "brain_ct", "histopathology", "retina", "skin", "ultrasound"]:
        class SubAdapter(MockAdapter): _MODALITY = mod
        reg.register_adapter(mod, SubAdapter)

    cfg = RoutingConfiguration()

    print("\n[1/5] Testing ROUTE_TO_MODEL outcome (Skin, 0.95 confidence)...")
    router_success = ModalityRouter(config=cfg, model_registry=reg, predictor=MockPredictor("skin", 0.95))
    ctx = EngineContext(request_id="req_001", session_id="sess_001", image=Image.new("RGB", (224, 224)))
    dec1 = router_success.route(ctx)
    print(f"      Outcome:     {dec1.outcome.value}")
    print(f"      Modality:    {dec1.detected_modality}")
    print(f"      Adapter:     {dec1.adapter_key}")
    assert dec1.outcome == RoutingOutcome.ROUTE_TO_MODEL

    print("\n[2/5] Testing LOW_CONFIDENCE outcome (Brain MRI, 0.75 confidence)...")
    router_low = ModalityRouter(config=cfg, model_registry=reg, predictor=MockPredictor("brain_mri", 0.75))
    ctx2 = EngineContext(request_id="req_002", session_id="sess_002", image=Image.new("RGB", (224, 224)))
    dec2 = router_low.route(ctx2)
    print(f"      Outcome:     {dec2.outcome.value}")
    print(f"      Reason:      {dec2.reason}")
    assert dec2.outcome == RoutingOutcome.LOW_CONFIDENCE

    print("\n[3/5] Testing UNSUPPORTED_MODALITY outcome (Dental Xray, 0.99 confidence)...")
    router_unsupp = ModalityRouter(config=cfg, model_registry=reg, predictor=MockPredictor("dental_xray", 0.99))
    ctx3 = EngineContext(request_id="req_003", session_id="sess_003", image=Image.new("RGB", (224, 224)))
    dec3 = router_unsupp.route(ctx3)
    print(f"      Outcome:     {dec3.outcome.value}")
    print(f"      Reason:      {dec3.reason}")
    assert dec3.outcome == RoutingOutcome.UNSUPPORTED_MODALITY

    print("\n[4/5] Testing MANUAL_MODALITY_OVERRIDE (Brain CT override)...")
    ctx4 = EngineContext(request_id="req_004", session_id="sess_004", manual_modality_override="brain_ct", image=Image.new("RGB", (224, 224)))
    dec4 = router_success.route(ctx4)
    print(f"      Outcome:     {dec4.outcome.value}")
    print(f"      Modality:    {dec4.detected_modality}")
    print(f"      Mode:        {dec4.routing_mode}")
    assert dec4.outcome == RoutingOutcome.ROUTE_TO_MODEL
    assert dec4.routing_mode == "manual_override"

    print("\n[5/5] Router Health Check...")
    health = router_success.health_check()
    print(f"      Status:      {health['status']}")
    print(f"      Supported:   {len(health['supported_modalities'])} modalities")

    print("\n" + "-" * 66)
    print("Running PyTest Suite...")
    print("-" * 66)

    ModelRegistry.reset_instance()

    ret = pytest.main([
        "medical_ai_engine/tests/test_router.py",
        "-v",
        "--tb=short",
    ])

    if ret == 0:
        print("\n[SUCCESS] All Module 14.2.3 Router tests PASSED!")
    else:
        print(f"\n[FAILED] PyTest exit code: {ret}")
        sys.exit(ret)


if __name__ == "__main__":
    main()
