"""
workflow_coordinator.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Coordinates post-prediction / post-explainability anomaly workflow.

Workflow Steps
--------------
Prediction Completed
      ↓
Explainability Completed (if applicable)
      ↓
Anomaly Detection Engine  [ExecutionManager]
      ↓
Decision Fusion          [DecisionEngine]
      ↓
AnomalyResult Attachment [PredictionResult.extra["anomaly_result"]]
      ↓
Enriched PredictionResult

Failure Recovery Policy
-----------------------
If the anomaly subsystem fails, the ORIGINAL PredictionResult (including any explainability
artifacts attached previously) is PRESERVED and returned intact.
The anomaly exception is logged and recorded under `prediction_result.extra["anomaly_error"]`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Optional

from .execution_manager import ExecutionManager
from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import WorkflowCoordinationError
from .integration_metrics import AnomalyTimings, IntegrationMetrics
from .integration_validator import IntegrationValidator

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Coordinates post-prediction / post-explainability anomaly detection execution.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        execution_manager: Optional[ExecutionManager] = None,
        validator: Optional[IntegrationValidator] = None,
        metrics: Optional[IntegrationMetrics] = None,
    ) -> None:
        self._config = config or IntegrationConfiguration.default()
        self._exec_mgr = execution_manager or ExecutionManager(config=self._config)
        self._validator = validator or IntegrationValidator(
            supported_modalities=self._config.supported_modalities
        )
        self._metrics = metrics or IntegrationMetrics()

    def coordinate(
        self,
        engine_ctx: Any,
        prediction_result: Any,
    ) -> Any:
        """
        Execute post-prediction / post-explainability anomaly detection.

        Attaches `AnomalyResult` to `prediction_result.extra["anomaly_result"]`.

        Parameters
        ----------
        engine_ctx : Any
            Context from MedicalAIEngine execution.
        prediction_result : Any
            PredictionResult from prediction (and optional explainability).

        Returns
        -------
        PredictionResult
            Enriched PredictionResult with anomaly_result attached.
            On error (when fail_silently=True), returns original prediction_result.
        """
        if not self._config.enabled:
            self._metrics.record_skip()
            return prediction_result

        t0 = time.perf_counter()

        try:
            self._validator.validate_inputs(engine_ctx, prediction_result)

            if not self._validator.is_eligible(prediction_result):
                self._metrics.record_skip()
                return prediction_result

            # Run execution pipeline
            anomaly_result, timings = self._exec_mgr.execute_pipeline(
                prediction_result=prediction_result,
                engine_ctx=engine_ctx,
            )

            # Record overall integration time
            total_integration_ms = (time.perf_counter() - t0) * 1000.0
            updated_timings = AnomalyTimings(
                strategy_execution_ms=timings.strategy_execution_ms,
                unknown_detection_ms=timings.unknown_detection_ms,
                decision_fusion_ms=timings.decision_fusion_ms,
                anomaly_total_latency_ms=timings.anomaly_total_latency_ms,
                overall_integration_ms=total_integration_ms,
            )
            self._metrics.record_run(updated_timings, success=True)

            # Attach AnomalyResult to prediction_result
            return self._attach_anomaly_result(prediction_result, anomaly_result)

        except Exception as exc:
            self._metrics.record_run(AnomalyTimings(overall_integration_ms=(time.perf_counter() - t0) * 1000.0), success=False)
            logger.warning("[WorkflowCoordinator] Anomaly integration failed: %s", exc, exc_info=True)
            if self._config.fail_silently:
                return self._attach_anomaly_error(prediction_result, str(exc))
            raise WorkflowCoordinationError(f"Workflow coordination failed: {exc}") from exc

    def _attach_anomaly_result(self, prediction_result: Any, anomaly_result: Any) -> Any:
        """Attach AnomalyResult dictionary to prediction_result.extra."""
        extra = dict(getattr(prediction_result, "extra", {}) or {})
        extra["anomaly_result"] = anomaly_result.to_dict()
        extra["anomaly_status"] = anomaly_result.status.value
        extra["risk_level"] = anomaly_result.risk_level.value
        extra["recommended_action"] = anomaly_result.recommended_action.value

        if hasattr(prediction_result, "with_extra"):
            return prediction_result.with_extra(extra)
        try:
            return replace(prediction_result, extra=extra)
        except Exception:
            setattr(prediction_result, "extra", extra)
            return prediction_result

    def _attach_anomaly_error(self, prediction_result: Any, error_msg: str) -> Any:
        """Attach anomaly_error string to prediction_result.extra on failure."""
        extra = dict(getattr(prediction_result, "extra", {}) or {})
        extra["anomaly_error"] = error_msg
        if hasattr(prediction_result, "with_extra"):
            return prediction_result.with_extra(extra)
        try:
            return replace(prediction_result, extra=extra)
        except Exception:
            setattr(prediction_result, "extra", extra)
            return prediction_result

    @property
    def metrics(self) -> IntegrationMetrics:
        return self._metrics


__all__ = ["WorkflowCoordinator"]
