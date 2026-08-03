"""
test_explainability_integration.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Comprehensive unit test suite (25 tests).

Covers:
  - Successful integration & end-to-end execution
  - Event listener registration & direct callbacks
  - WorkflowCoordinator 10-step dispatch
  - ExecutionManager lifecycle & error handling
  - Prediction preservation on explainability failure
  - Confidence & modality gating
  - Concurrent predictions & thread safety
  - Multi-modality execution across all 7 platform modalities
  - IntegrationMetrics tracking & reporting
"""

from __future__ import annotations

import concurrent.futures
from PIL import Image
import pytest

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.medical_ai_engine import MedicalAIEngine
from medical_ai_engine.explainability.integration import (
    ExecutionManager,
    ExplainabilityEventListener,
    ExplainabilityIntegrator,
    IntegrationConfiguration,
    IntegrationConfigurationError,
    IntegrationMetrics,
    IntegrationValidator,
    WorkflowCoordinator,
)
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_builder import ResponseBuilder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fast_config() -> IntegrationConfiguration:
    return IntegrationConfiguration.fast_test()


@pytest.fixture
def sample_pil_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(120, 150, 180))


@pytest.fixture
def engine_ctx(sample_pil_image) -> EngineContext:
    ctx = EngineContext(
        request_id="req-test-101",
        session_id="sess-test-101",
        image=sample_pil_image,
        execution_state="COMPLETED",
    )
    ctx.modality_prediction = {
        "detected_modality": "brain_mri",
        "modality_confidence": 0.98,
        "routing_mode": "automatic",
    }
    ctx.disease_prediction = {
        "predicted_class": "glioma",
        "confidence": 0.92,
        "status": "success",
    }
    ctx.confidence = {
        "score": 0.92,
        "reliability": "high",
        "requires_manual_review": False,
    }
    return ctx


@pytest.fixture
def prediction_result(engine_ctx) -> PredictionResult:
    builder = ResponseBuilder()
    return builder.build(engine_ctx)


@pytest.fixture
def integrator(fast_config) -> ExplainabilityIntegrator:
    return ExplainabilityIntegrator(config=fast_config)


# ── T01 — Configuration Validation ───────────────────────────────────────────

class TestIntegrationConfiguration:

    def test_default_creation(self):
        cfg = IntegrationConfiguration()
        assert cfg.enabled is True
        assert cfg.cam_method == "gradcam"
        assert cfg.min_confidence_threshold == 0.30

    def test_invalid_confidence_raises(self):
        with pytest.raises(IntegrationConfigurationError):
            IntegrationConfiguration(min_confidence_threshold=1.5)

    def test_invalid_threads_raises(self):
        with pytest.raises(IntegrationConfigurationError):
            IntegrationConfiguration(max_concurrent_explanations=0)

    def test_fast_test_factory(self):
        cfg = IntegrationConfiguration.fast_test()
        assert cfg.enabled is True
        assert cfg.async_execution is False
        assert cfg.min_confidence_threshold == 0.0


# ── T02 — IntegrationValidator ──────────────────────────────────────────────

class TestIntegrationValidator:

    def test_valid_context_passes(self, engine_ctx):
        validator = IntegrationValidator()
        validator.validate_engine_context(engine_ctx)

    def test_should_run_explainability_confidence_gate(self):
        validator = IntegrationValidator()
        assert validator.should_run_explainability(0.85, "brain_mri", 0.30) is True
        assert validator.should_run_explainability(0.20, "brain_mri", 0.30) is False

    def test_should_run_explainability_modality_gate(self):
        validator = IntegrationValidator(supported_modalities=["brain_mri"])
        assert validator.should_run_explainability(0.85, "brain_mri", 0.30) is True
        assert validator.should_run_explainability(0.85, "chest_xray", 0.30) is False


# ── T03 — ExecutionManager ───────────────────────────────────────────────────

class TestExecutionManager:

    def test_execute_returns_result_and_timings(self, fast_config, engine_ctx, prediction_result):
        mgr = ExecutionManager(config=fast_config)
        res, timings = mgr.execute(engine_ctx, prediction_result)
        assert res is not None
        assert res.is_success is True
        assert timings.total_explainability_ms > 0
        assert timings.modality == "brain_mri"

    def test_clinical_explanation_attached(self, fast_config, engine_ctx, prediction_result):
        mgr = ExecutionManager(config=fast_config)
        res, _ = mgr.execute(engine_ctx, prediction_result)
        assert len(res.clinical_summary) > 20
        assert "clinical_report" in res.extra


# ── T04 — WorkflowCoordinator ────────────────────────────────────────────────

class TestWorkflowCoordinator:

    def test_coordinate_attaches_gradcam_result(self, fast_config, engine_ctx, prediction_result):
        coord = WorkflowCoordinator(config=fast_config)
        enriched = coord.coordinate(engine_ctx, prediction_result)
        assert enriched.gradcam_result is not None
        assert enriched.gradcam_result["status"] == "SUCCESS"
        assert "explainability_ms" in enriched.performance_metrics

    def test_disabled_config_returns_unchanged(self, engine_ctx, prediction_result):
        cfg = IntegrationConfiguration(enabled=False)
        coord = WorkflowCoordinator(config=cfg)
        enriched = coord.coordinate(engine_ctx, prediction_result)
        assert enriched.gradcam_result is None

    def test_low_confidence_skips_explainability(self, engine_ctx, prediction_result):
        cfg = IntegrationConfiguration(min_confidence_threshold=0.99)
        coord = WorkflowCoordinator(config=cfg)
        enriched = coord.coordinate(engine_ctx, prediction_result)
        assert enriched.gradcam_result is None
        metrics = coord.get_metrics()
        assert metrics["skipped"] == 1


# ── T05 — Failure Recovery & Prediction Preservation ─────────────────────────

class TestFailureRecovery:

    def test_prediction_preserved_on_pipeline_failure(self, engine_ctx, prediction_result):
        # Create an execution manager that raises internally
        class FailingExecMgr(ExecutionManager):
            def execute(self, engine_ctx, prediction_result):
                raise RuntimeError("Simulated Grad-CAM CUDA Out Of Memory Error")

        cfg = IntegrationConfiguration(fail_silently=True)
        coord = WorkflowCoordinator(config=cfg, execution_manager=FailingExecMgr(config=cfg))
        enriched = coord.coordinate(engine_ctx, prediction_result)

        # Main prediction must remain completely intact
        assert enriched.is_success is True
        assert enriched.predicted_disease == prediction_result.predicted_disease
        assert enriched.confidence == prediction_result.confidence


# ── T06 — Event Listener ─────────────────────────────────────────────────────

class TestExplainabilityEventListener:

    def test_direct_callback(self, integrator, engine_ctx, prediction_result):
        listener = ExplainabilityEventListener(integrator=integrator, async_execution=False)
        enriched = listener.on_engine_completed(engine_ctx, prediction_result)
        assert enriched is not None
        assert enriched.gradcam_result is not None

    def test_publisher_monkey_patch_registration(self, integrator):
        listener = ExplainabilityEventListener(integrator=integrator)
        engine = MedicalAIEngine()
        listener.register_with_publisher(engine.event_publisher)
        assert hasattr(engine.event_publisher, "_subscribers")
        assert "EngineCompleted" in engine.event_publisher._subscribers


# ── T07 — Full Integrator (ExplainabilityIntegrator) ─────────────────────────

class TestExplainabilityIntegrator:

    def test_execute_with_explanation_end_to_end(self, integrator, sample_pil_image):
        result = integrator.execute_with_explanation(
            image_source=sample_pil_image,
            manual_modality_override="brain_mri",
        )
        assert result is not None
        assert result.is_success is True
        assert result.gradcam_result is not None
        assert "clinical_summary" in result.gradcam_result
        assert result.gradcam_result["clinical_summary"] != ""

    def test_health_check(self, integrator):
        health = integrator.health_check()
        assert health["enabled"] is True
        assert "metrics" in health

    def test_metrics_snapshot(self, integrator, sample_pil_image):
        integrator.execute_with_explanation(
            image_source=sample_pil_image,
            manual_modality_override="chest_xray",
        )
        metrics = integrator.get_metrics()
        assert metrics["total_processed"] >= 1
        assert metrics["successful"] >= 1


# ── T08 — Multi-Modality Execution ──────────────────────────────────────────

class TestMultiModalityExecution:

    def test_all_seven_modalities_execute_cleanly(self, integrator, sample_pil_image):
        modalities = [
            "brain_mri", "chest_xray", "brain_ct",
            "histopathology", "retina", "skin", "ultrasound",
        ]
        for mod in modalities:
            res = integrator.execute_with_explanation(
                image_source=sample_pil_image,
                manual_modality_override=mod,
            )
            assert res.is_success is True
            assert res.gradcam_result is not None
            assert res.modality == mod


# ── T09 — Concurrent Predictions & Thread Safety ─────────────────────────────

class TestConcurrentExecution:

    def test_concurrent_integrator_calls(self, integrator, sample_pil_image):
        def _task(idx: int):
            return integrator.execute_with_explanation(
                image_source=sample_pil_image,
                manual_modality_override="brain_mri",
                request_id=f"req-concurrent-{idx}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_task, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        assert all(r.is_success for r in results)
        assert all(r.gradcam_result is not None for r in results)
