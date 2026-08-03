"""
test_anomaly_integration.py — Unit & Integration tests for Module 16.6 Anomaly Engine Integration.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pytest

from medical_ai_engine.anomaly.integration import (
    AnomalyEventListener,
    AnomalyIntegrator,
    AnomalyTimings,
    ExecutionManager,
    IntegrationConfiguration,
    IntegrationConfigurationError,
    IntegrationMetrics,
    IntegrationValidator,
    WorkflowCoordinator,
)


@dataclass
class DummyPredictionResult:
    request_id: str = "req-int-101"
    session_id: str = "sess-int-101"
    modality: str = "chest_xray"
    predicted_disease: str = "pneumonia"
    confidence: float = 0.94
    extra: Dict[str, Any] = field(default_factory=lambda: {"gradcam_result": {"heatmap": "fake_map"}})

    def with_extra(self, new_extra: Dict[str, Any]) -> "DummyPredictionResult":
        return DummyPredictionResult(
            request_id=self.request_id,
            session_id=self.session_id,
            modality=self.modality,
            predicted_disease=self.predicted_disease,
            confidence=self.confidence,
            extra=new_extra,
        )


@dataclass
class DummyEngineContext:
    request_id: str = "req-int-101"
    session_id: str = "sess-int-101"


# ----------------------------------------------------------------------
# Configuration & Metrics Tests
# ----------------------------------------------------------------------
class TestIntegrationConfigAndMetrics:
    def test_default_config(self):
        cfg = IntegrationConfiguration.default()
        assert cfg.enabled is True
        assert cfg.fail_silently is True
        assert "chest_xray" in cfg.supported_modalities

    def test_invalid_config_raises(self):
        with pytest.raises(IntegrationConfigurationError):
            IntegrationConfiguration(timeout_ms=-50.0)

    def test_metrics_recording(self):
        metrics = IntegrationMetrics()
        timings = AnomalyTimings(
            strategy_execution_ms=5.0,
            unknown_detection_ms=10.0,
            decision_fusion_ms=3.0,
            anomaly_total_latency_ms=18.0,
            overall_integration_ms=20.0,
        )
        metrics.record_run(timings, success=True)
        summary = metrics.get_summary()

        assert summary["total_requests"] == 1
        assert summary["successful_runs"] == 1
        assert summary["avg_integration_latency_ms"] == 20.0


# ----------------------------------------------------------------------
# Validator Tests
# ----------------------------------------------------------------------
class TestIntegrationValidator:
    def test_eligibility_check(self):
        validator = IntegrationValidator(supported_modalities=["chest_xray", "brain_mri"])
        pred_valid = DummyPredictionResult(modality="chest_xray")
        pred_invalid = DummyPredictionResult(modality="unsupported_modality")

        assert validator.is_eligible(pred_valid) is True
        assert validator.is_eligible(pred_invalid) is False


# ----------------------------------------------------------------------
# ExecutionManager Tests
# ----------------------------------------------------------------------
class TestExecutionManager:
    def test_execute_pipeline(self):
        exec_mgr = ExecutionManager()
        pred = DummyPredictionResult()
        ctx = DummyEngineContext()

        result, timings = exec_mgr.execute_pipeline(pred, ctx)
        assert result.request_id == "req-int-101"
        assert result.status is not None
        assert result.risk_level is not None
        assert timings.overall_integration_ms >= 0.0


# ----------------------------------------------------------------------
# End-to-End WorkflowCoordinator Integration Tests
# ----------------------------------------------------------------------
class TestWorkflowCoordinator:
    def test_end_to_end_anomaly_workflow(self):
        """
        Verify complete workflow:
        Prediction Completed -> Explainability Completed -> Anomaly Detection -> AnomalyResult attached.
        """
        coordinator = WorkflowCoordinator()
        pred = DummyPredictionResult()
        ctx = DummyEngineContext()

        enriched = coordinator.coordinate(ctx, pred)

        assert "anomaly_result" in enriched.extra
        assert "gradcam_result" in enriched.extra  # Explainability preserved!
        anomaly = enriched.extra["anomaly_result"]
        assert anomaly["status"] in ("KNOWN_DISEASE", "UNKNOWN_DISEASE", "OUT_OF_DISTRIBUTION", "LOW_CONFIDENCE")
        assert "risk_level" in anomaly
        assert "recommended_action" in anomaly

    def test_failure_recovery_preserves_prediction_and_explainability(self):
        """
        Verify that if anomaly execution encounters a failure, fail_silently=True ensures
        the original prediction_result and explainability artifacts are preserved untouched.
        """
        cfg = IntegrationConfiguration(fail_silently=True)

        class BrokenExecutionManager(ExecutionManager):
            def execute_pipeline(self, *args, **kwargs):
                raise RuntimeError("Simulated anomaly failure")

        coordinator = WorkflowCoordinator(config=cfg, execution_manager=BrokenExecutionManager())
        pred = DummyPredictionResult()
        ctx = DummyEngineContext()

        enriched = coordinator.coordinate(ctx, pred)

        # Prediction and Explainability are preserved!
        assert enriched.predicted_disease == "pneumonia"
        assert "gradcam_result" in enriched.extra
        assert "anomaly_error" in enriched.extra
        assert "Simulated anomaly failure" in enriched.extra["anomaly_error"]

    def test_disabled_integration_bypasses(self):
        cfg = IntegrationConfiguration(enabled=False)
        coordinator = WorkflowCoordinator(config=cfg)
        pred = DummyPredictionResult()
        ctx = DummyEngineContext()

        res = coordinator.coordinate(ctx, pred)
        assert "anomaly_result" not in res.extra


# ----------------------------------------------------------------------
# AnomalyIntegrator & EventListener Tests
# ----------------------------------------------------------------------
class TestAnomalyIntegratorAndListener:
    def test_facade_process_completed_prediction(self):
        integrator = AnomalyIntegrator()
        pred = DummyPredictionResult()
        ctx = DummyEngineContext()

        enriched = integrator.process_completed_prediction(ctx, pred)
        assert "anomaly_result" in enriched.extra

    def test_event_listener_registration(self):
        integrator = AnomalyIntegrator()
        listener = AnomalyEventListener(integrator=integrator)

        class DummyPublisher:
            pass

        pub = DummyPublisher()
        listener.register_with_publisher(pub)

        assert hasattr(pub, "_subscribers")
        assert "EngineCompletedEvent" in pub._subscribers

    def test_concurrent_execution(self):
        integrator = AnomalyIntegrator()

        def run_request(i: int):
            pred = DummyPredictionResult(request_id=f"req-conc-{i}")
            ctx = DummyEngineContext(request_id=f"req-conc-{i}")
            return integrator.process_completed_prediction(ctx, pred)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_request, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for r in results:
            assert "anomaly_result" in r.extra
