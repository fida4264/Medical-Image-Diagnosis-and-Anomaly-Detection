"""
Medical AI Engine Pipeline — Test Runner Script (Module 14.2.4)

Demonstrates:
  1. Default 9-Stage Pipeline Execution
  2. EngineContext Progress Transformation
  3. Reverse-order Rollback on Failure
  4. Full PyTest Suite Execution
"""

import sys
import pytest
from PIL import Image

from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.medical_ai_engine import MedicalAIEngine
from medical_ai_engine.pipeline.pipeline_exceptions import PipelineExecutionException
from medical_ai_engine.pipeline.pipeline_executor import PipelineExecutor
from medical_ai_engine.pipeline.stages import AbstractPipelineStage
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.routing.modality_router import ModalityRouter
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from modality_classifier.inference.predictor import ModalityPredictionResult


class DemoPredictor:
    def predict(self, image_input, top_k=3):
        return ModalityPredictionResult(
            predicted_modality="chest_xray",
            confidence=0.96,
            top_k_predictions=[{"modality": "chest_xray", "confidence": 0.96}],
            inference_time_ms=15.0,
            routing_destination="Module 8 (Chest X-ray)",
        )


class DemoChestXrayAdapter(BasePredictorAdapter):
    _MODALITY = "chest_xray"
    @property
    def modality(self): return self._MODALITY
    def load_model(self, checkpoint_path=None, label_map_path=None): self._is_loaded = True
    def predict(self, image):
        return {"predicted_class": "pneumonia", "confidence": 0.96, "requires_manual_review": False}


def main():
    print("=" * 66)
    print("      Module 14.2.4: Orchestration Pipeline Stages Runner     ")
    print("=" * 66)

    ModelRegistry.reset_instance()
    reg = ModelRegistry.get_instance(config=RegistryConfiguration(auto_discover_adapters=False))
    reg.register_adapter("chest_xray", DemoChestXrayAdapter)

    print("\n[1/4] Running 9-Stage Pipeline via MedicalAIEngine Core...")
    mock_router = ModalityRouter(
        config=RoutingConfiguration(),
        model_registry=reg,
        predictor=DemoPredictor(),
    )

    # Instantiate engine which defaults to 9-stage pipeline
    engine = MedicalAIEngine()
    engine.initialize()

    img = Image.new("RGB", (224, 224), color=(0, 255, 0))
    # Replace router inside stage 3
    engine.pipeline_stages[2]._router = mock_router

    ctx = engine.execute(image_source=img, request_id="req_pipeline_demo_1")

    print(f"      Execution State:   {ctx.execution_state}")
    print(f"      Detected Modality: {ctx.modality_prediction.get('detected_modality')}")
    print(f"      Diagnosis:         {ctx.disease_prediction.get('predicted_class')}")
    print(f"      Reliability:       {ctx.confidence.get('reliability')}")
    print(f"      Report Type:       {ctx.clinical_report.get('report_type')}")
    print(f"      Total Time:        {ctx.performance_metrics.get('total_pipeline_ms'):.2f}ms")
    assert ctx.execution_state == "COMPLETED"

    print("\n[2/4] Verifying EngineContext State Mutation...")
    assert ctx.image is not None
    assert ctx.prediction_result["status"] == "SUCCESS"
    print("      EngineContext fields populated cleanly.")

    print("\n[3/4] Testing Rollback on Stage Failure...")
    class FailStage(AbstractPipelineStage):
        def __init__(self): super().__init__("FailingStageDemo")
        def process(self, ctx): raise ValueError("Simulated Stage Failure")

    roll_events = []
    class TrackingStage(AbstractPipelineStage):
        def __init__(self, name): super().__init__(name)
        def process(self, ctx): pass
        def rollback(self, ctx): roll_events.append(self.name)

    fail_executor = PipelineExecutor(stages=[TrackingStage("StageA"), TrackingStage("StageB"), FailStage()])
    fail_ctx = EngineContext(request_id="req_fail_demo", session_id="sess_fail_demo")

    try:
        fail_executor.execute(fail_ctx)
    except PipelineExecutionException:
        print(f"      Caught expected PipelineExecutionException.")
        print(f"      Rollback order executed: {roll_events}")
        assert roll_events == ["StageB", "StageA"]

    print("\n[4/4] Running PyTest Pipeline Test Suite...")
    print("-" * 66)

    ModelRegistry.reset_instance()

    ret = pytest.main([
        "medical_ai_engine/tests/test_pipeline.py",
        "-v",
        "--tb=short",
    ])

    if ret == 0:
        print("\n[SUCCESS] All Module 14.2.4 Pipeline tests PASSED!")
    else:
        print(f"\n[FAILED] PyTest exit code: {ret}")
        sys.exit(ret)


if __name__ == "__main__":
    main()
