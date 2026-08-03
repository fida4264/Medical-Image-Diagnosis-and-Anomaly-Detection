"""
Medical AI Engine — Pipeline Executor & Stages Test Suite (Module 14.2.4)

Tests:
  - Default 9-stage pipeline creation & execution order
  - Shared EngineContext mutation across all stages
  - Rollback behavior in REVERSE order on stage failure
  - Event publication (StageStartedEvent, StageCompletedEvent, StageFailedEvent, StageRollbackEvent)
  - Performance metrics and stage timing collection
  - Pipeline failure handling & exception propagation
  - Concurrent pipeline execution (20 threads)
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
import pytest
from PIL import Image

from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.events.engine_events import EngineEventPublisher
from medical_ai_engine.events.stage_events import (
    StageCompletedEvent,
    StageFailedEvent,
    StageRollbackEvent,
    StageStartedEvent,
)
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage
from medical_ai_engine.pipeline.pipeline_exceptions import PipelineExecutionException, StageValidationException
from medical_ai_engine.pipeline.pipeline_executor import PipelineExecutor
from medical_ai_engine.pipeline.pipeline_stage_factory import PipelineStageFactory
from medical_ai_engine.pipeline.stages import (
    AdapterPredictionStage,
    ClinicalReportHookStage,
    ConfidenceAssessmentStage,
    ImageQualityAssessmentStage,
    InputValidationStage,
    LoggingStage,
    ModalityClassificationStage,
    ResponsePreparationStage,
    RoutingStage,
)
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.routing.modality_router import ModalityRouter
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from modality_classifier.inference.predictor import ModalityPredictionResult


# ── Mocks ──────────────────────────────────────────────────────────────────────

class MockPredictor:
    """Mock Module 6 predictor for pipeline test."""
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


class MockSkinAdapter(BasePredictorAdapter):
    _MODALITY = "skin"
    @property
    def modality(self) -> str: return self._MODALITY
    def load_model(self, checkpoint_path=None, label_map_path=None): self._is_loaded = True
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        return {
            "modality": "skin",
            "predicted_class": "melanoma",
            "confidence": 0.94,
            "requires_manual_review": False,
        }


class FailingStage(BasePipelineStage):
    """Mock stage that intentionally fails process()."""
    @property
    def name(self) -> str: return "FailingStage"
    def process(self, ctx: EngineContext) -> None:
        raise RuntimeError("Simulated stage processing error")


class MockTrackingStage(BasePipelineStage):
    """Mock stage tracking execution and rollback."""
    def __init__(self, stage_name: str, execution_log: list, rollback_log: list):
        self._name = stage_name
        self.exec_log = execution_log
        self.roll_log = rollback_log

    @property
    def name(self) -> str: return self._name

    def process(self, ctx: EngineContext) -> None:
        self.exec_log.append(self.name)

    def rollback(self, ctx: EngineContext) -> None:
        self.roll_log.append(self.name)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_environment():
    ModelRegistry.reset_instance()
    reg = ModelRegistry.get_instance(config=RegistryConfiguration(auto_discover_adapters=False))
    reg.register_adapter("skin", MockSkinAdapter)
    yield
    ModelRegistry.reset_instance()


def build_pipeline_executor() -> PipelineExecutor:
    mock_router = ModalityRouter(
        config=RoutingConfiguration(),
        model_registry=ModelRegistry.get_instance(),
        predictor=MockPredictor(modality="skin", confidence=0.95),
    )

    stages = [
        InputValidationStage(),
        ImageQualityAssessmentStage(),
        ModalityClassificationStage(router=mock_router),
        RoutingStage(),
        AdapterPredictionStage(),
        ConfidenceAssessmentStage(),
        ClinicalReportHookStage(),
        ResponsePreparationStage(),
        LoggingStage(),
    ]

    return PipelineExecutor(stages=stages)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_factory_default_stages():
    stages = PipelineStageFactory.create_default_pipeline()
    assert len(stages) == 9
    stage_names = [s.name for s in stages]
    assert stage_names == [
        "InputValidationStage",
        "ImageQualityAssessmentStage",
        "ModalityClassificationStage",
        "RoutingStage",
        "AdapterPredictionStage",
        "ConfidenceAssessmentStage",
        "ClinicalReportHookStage",
        "ResponsePreparationStage",
        "LoggingStage",
    ]


def test_full_pipeline_execution_success():
    executor = build_pipeline_executor()
    ctx = EngineContext(
        request_id="req_full_001",
        session_id="sess_full_001",
        image=Image.new("RGB", (224, 224), color=(255, 0, 0)),
    )

    out_ctx = executor.execute(ctx)

    assert out_ctx is ctx  # Identical reference
    assert out_ctx.modality_prediction["detected_modality"] == "skin"
    assert out_ctx.disease_prediction["predicted_class"] == "melanoma"
    assert out_ctx.confidence["reliability"] == "high"
    assert out_ctx.clinical_report["primary_diagnosis"] == "melanoma"
    assert out_ctx.prediction_result["status"] == "SUCCESS"
    assert "total_pipeline_ms" in out_ctx.performance_metrics


def test_stage_execution_order_and_rollback_reverse():
    exec_log = []
    roll_log = []

    s1 = MockTrackingStage("Stage1", exec_log, roll_log)
    s2 = MockTrackingStage("Stage2", exec_log, roll_log)
    s3 = FailingStage()

    executor = PipelineExecutor(stages=[s1, s2, s3])
    ctx = EngineContext(request_id="req_fail_001", session_id="sess_fail_001")

    with pytest.raises(PipelineExecutionException) as exc_info:
        executor.execute(ctx)

    assert "FailingStage" in str(exc_info.value)
    assert exec_log == ["Stage1", "Stage2"]
    # Rollback must occur in REVERSE order
    assert roll_log == ["Stage2", "Stage1"]
    assert len(ctx.errors) == 1
    assert ctx.errors[0]["stage"] == "FailingStage"


def test_event_publication_during_pipeline():
    publisher = EngineEventPublisher()
    executor = build_pipeline_executor()
    executor.event_publisher = publisher

    ctx = EngineContext(
        request_id="req_event_001",
        session_id="sess_event_001",
        image=Image.new("RGB", (224, 224)),
    )

    executor.execute(ctx)
    events = publisher.get_published_events()

    event_types = [e["event_type"] for e in events]
    assert "StageStartedEvent" in event_types
    assert "StageCompletedEvent" in event_types

    # Ensure started and completed were published for each of the 9 stages
    started_count = sum(1 for e in events if e["event_type"] == "StageStartedEvent")
    completed_count = sum(1 for e in events if e["event_type"] == "StageCompletedEvent")
    assert started_count == 9
    assert completed_count == 9


def test_performance_metrics_collection():
    executor = build_pipeline_executor()
    ctx = EngineContext(
        request_id="req_metric_001",
        session_id="sess_metric_001",
        image=Image.new("RGB", (224, 224)),
    )

    executor.execute(ctx)
    metrics = ctx.performance_metrics

    assert "InputValidationStage_ms" in metrics
    assert "RoutingStage_ms" in metrics
    assert "AdapterPredictionStage_ms" in metrics
    assert "total_pipeline_ms" in metrics
    assert metrics["total_pipeline_ms"] >= 0.0


def test_concurrent_pipeline_execution():
    executor = build_pipeline_executor()
    results = []
    errors = []

    def worker(idx: int):
        try:
            ctx = EngineContext(
                request_id=f"req_thread_{idx}",
                session_id=f"sess_thread_{idx}",
                image=Image.new("RGB", (224, 224)),
            )
            out = executor.execute(ctx)
            results.append(out)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 20
    assert all(r.prediction_result["status"] == "SUCCESS" for r in results)
