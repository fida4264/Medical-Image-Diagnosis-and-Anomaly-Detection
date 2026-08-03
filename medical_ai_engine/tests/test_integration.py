"""
Module 14.2.6: End-to-End System Integration Test Suite

Tests:
  1. All 7 Disease Modality Workflows (Brain MRI, Chest X-ray, Brain CT, Histopathology, Retina, Skin, Ultrasound)
  2. Failure Scenarios (unsupported modality, low confidence, missing adapter, pipeline failure)
  3. Integration Metrics & Performance Breakdown
  4. Concurrent 20-thread Workflow Execution
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict, List
import pytest
from PIL import Image

from medical_ai_engine.integration.engine_integrator import EngineIntegrator
from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
from medical_ai_engine.integration.integration_validator import IntegrationValidator
from medical_ai_engine.integration.integration_metrics import IntegrationMetrics
from medical_ai_engine.integration.integration_exceptions import (
    WorkflowExecutionException,
    IntegrationValidationException,
)
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_schema import PredictionStatus


# ── Helper: create test image ──────────────────────────────────────────────────

def _make_dummy_pil_image() -> Image.Image:
    """Create a 128x128 RGB test image in memory."""
    return Image.new("RGB", (128, 128), color=(100, 150, 200))


@pytest.fixture(scope="module")
def integrator() -> EngineIntegrator:
    """Fixture providing initialized EngineIntegrator with fallback adapters."""
    cfg = IntegrationConfiguration.test_config()
    integrator = EngineIntegrator(config=cfg)
    return integrator


# ── 1. All 7 Disease Modality Workflow Tests ───────────────────────────────────

class TestAllModalityWorkflows:

    @pytest.mark.parametrize(
        "modality_key",
        [
            "brain_mri",
            "chest_xray",
            "brain_ct",
            "histopathology",
            "retina",
            "skin",
            "ultrasound",
        ],
    )
    def test_end_to_end_modality_workflow(self, integrator: EngineIntegrator, modality_key: str):
        """Verify that each of the 7 supported modalities traverses the engine correctly."""
        img = _make_dummy_pil_image()
        executor = WorkflowExecutor(integrator=integrator)

        result, serialized, ctx = executor.execute(
            image_source=img,
            manual_modality_override=modality_key,
            format="dict",
            expected_modality=modality_key,
        )

        assert isinstance(result, PredictionResult)
        assert result.modality == modality_key
        assert result.status in (PredictionStatus.SUCCESS, PredictionStatus.COMPLETED_WITH_WARNINGS)
        assert result.predicted_disease != "unknown"
        assert result.confidence > 0.0
        assert "primary_diagnosis" in result.clinical_report
        assert "recommendations" in result.clinical_report
        assert isinstance(serialized, dict)
        assert serialized["modality"] == modality_key
        assert ctx.execution_state == "COMPLETED"


# ── 2. Failure Scenarios ───────────────────────────────────────────────────────

class TestFailureScenarios:

    def test_unsupported_modality_recovery(self, integrator: EngineIntegrator):
        """Manual override with unknown modality should be caught gracefully."""
        img = _make_dummy_pil_image()
        # Use config that catches pipeline exceptions
        cfg = IntegrationConfiguration(
            auto_register_adapters=False,
            catch_pipeline_exceptions=True,
            enable_strict_validation=False,
        )
        executor = WorkflowExecutor(integrator=EngineIntegrator(config=cfg))

        result, serialized, ctx = executor.execute(
            image_source=img,
            manual_modality_override="unsupported_quantum_scan",
            format="dict",
        )

        assert result.status in (PredictionStatus.FAILED, PredictionStatus.COMPLETED_WITH_WARNINGS)
        assert isinstance(serialized, dict)

    def test_missing_adapter_graceful_recovery(self):
        """Request for modality with no registered adapter should produce FAILED or unhandled status gracefully."""
        img = _make_dummy_pil_image()
        cfg = IntegrationConfiguration(
            auto_register_adapters=False,
            use_mock_fallbacks_if_missing=False,
            catch_pipeline_exceptions=True,
            enable_strict_validation=False,
        )
        integrator = EngineIntegrator(config=cfg)
        executor = WorkflowExecutor(integrator=integrator)

        result, serialized, ctx = executor.execute(
            image_source=img,
            manual_modality_override="unregistered_modality",
            format="dict",
        )

        assert result.status in (PredictionStatus.FAILED, PredictionStatus.COMPLETED_WITH_WARNINGS)

    def test_validation_failure_detection(self, integrator: EngineIntegrator):
        """IntegrationValidator must catch ID mismatch or bad metrics."""
        validator = IntegrationValidator(IntegrationConfiguration(enable_strict_validation=False))
        dummy_result = PredictionResult(
            request_id="req-A",
            session_id="sess-B",  # Mismatch
            status=PredictionStatus.SUCCESS,
            modality="brain_mri",
            predicted_disease="glioma",
            confidence=0.95,
        )
        from medical_ai_engine.core.engine_context import EngineContext
        dummy_ctx = EngineContext(request_id="req-DIFF", session_id="sess-B")

        errors = validator.get_validation_errors(dummy_ctx, dummy_result)
        assert any("Request ID mismatch" in e for e in errors)


# ── 3. Integration Metrics & Performance Breakdown ────────────────────────────

class TestIntegrationMetrics:

    def test_metrics_collection_after_workflows(self, integrator: EngineIntegrator):
        metrics = IntegrationMetrics()
        executor = WorkflowExecutor(integrator=integrator, metrics=metrics)
        img = _make_dummy_pil_image()

        for modality in ["brain_mri", "chest_xray", "skin"]:
            executor.execute(image_source=img, manual_modality_override=modality)

        summary = metrics.get_summary()
        assert summary["total_workflows"] == 3
        assert summary["successful_workflows"] == 3
        assert summary["failed_workflows"] == 0
        assert summary["success_rate_pct"] == 100.0
        assert summary["avg_total_execution_ms"] > 0.0
        assert "brain_mri" in summary["workflows_per_modality"]


# ── 4. Concurrent 20-thread Workflow Execution ─────────────────────────────────

class TestConcurrentWorkflows:

    def test_concurrent_20_thread_workflows(self, integrator: EngineIntegrator):
        executor = WorkflowExecutor(integrator=integrator)
        results: List[PredictionResult] = []
        errors: List[Exception] = []

        modalities = [
            "brain_mri", "chest_xray", "brain_ct", "histopathology",
            "retina", "skin", "ultrasound",
        ]

        def run_one(thread_id: int):
            try:
                img = _make_dummy_pil_image()
                target_mod = modalities[thread_id % len(modalities)]
                res, _, _ = executor.execute(
                    image_source=img,
                    manual_modality_override=target_mod,
                    format="dict",
                    expected_modality=target_mod,
                )
                results.append(res)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent workflow errors: {errors}"
        assert len(results) == 20
        request_ids = {r.request_id for r in results}
        assert len(request_ids) == 20, "All 20 request_ids must be unique"
