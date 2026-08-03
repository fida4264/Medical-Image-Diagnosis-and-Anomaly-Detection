"""
workflow_coordinator.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Coordinates the post-prediction explainability dispatch workflow.

Responsibility
--------------
WorkflowCoordinator sits between WorkflowExecutor (Module 14.2.6) and the
ExplainabilityIntegrator. It wraps each step of the 10-point workflow defined
in the module specification:

  1.  Receive PredictionCompleted event (via direct call from ExplainabilityIntegrator)
  2.  Build ExplainabilityContext                 [→ ExecutionManager]
  3.  Select explainability method via CAM Registry [→ ExplainabilityPipeline]
  4.  Execute Grad-CAM                             [→ ExplainabilityPipeline]
  5.  Generate normalized heatmap                  [→ ExplainabilityPipeline]
  6.  Generate overlay visualization               [→ ExplainabilityPipeline]
  7.  Execute Clinical Explanation Engine          [→ ExecutionManager → ClinicalExplanationEngine]
  8.  Build ExplainabilityResult                   [→ ExplainabilityPipeline]
  9.  Attach ExplainabilityResult to PredictionResult [→ self._attach]
  10. Return unified platform response             [→ caller]

Failure policy
--------------
If any step fails, the ORIGINAL PredictionResult is returned unchanged.
The explanation error is logged and recorded in PredictionResult.extra.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Optional, Tuple

from .execution_manager import ExecutionManager
from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import WorkflowCoordinationError
from .integration_metrics import ExplainabilityTimings, IntegrationMetrics
from .integration_validator import IntegrationValidator

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Coordinates the 10-step post-prediction explainability dispatch workflow.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        execution_manager: Optional[ExecutionManager] = None,
        validator: Optional[IntegrationValidator] = None,
        metrics: Optional[IntegrationMetrics] = None,
    ) -> None:
        self._config = config or IntegrationConfiguration.fast_test()
        self._exec_mgr = execution_manager or ExecutionManager(config=self._config)
        self._validator = validator or IntegrationValidator(
            supported_modalities=self._config.supported_modalities
        )
        self._metrics = metrics or IntegrationMetrics()

    # ── Public API ────────────────────────────────────────────────────────────

    def coordinate(
        self,
        engine_ctx: Any,
        prediction_result: Any,
    ) -> Any:
        """
        Execute the 10-step explainability workflow and return an enriched
        PredictionResult with explainability artifacts attached.

        Parameters
        ----------
        engine_ctx:
            EngineContext returned by MedicalAIEngine.execute().
        prediction_result:
            PredictionResult built by ResponseBuilder (frozen dataclass).

        Returns
        -------
        Enriched PredictionResult with gradcam_result, clinical_summary, etc.
        On failure (if fail_silently=True), returns original prediction_result unchanged.
        """
        if not self._config.enabled:
            return prediction_result

        request_id = getattr(engine_ctx, "request_id", "unknown")
        modality    = getattr(prediction_result, "modality", "unknown")
        confidence  = float(getattr(prediction_result, "confidence", 0.0))

        # Step 1: Confidence + modality gate
        if not self._validator.should_run_explainability(
            confidence=confidence,
            modality=modality,
            threshold=self._config.min_confidence_threshold,
            supported_modalities=self._config.supported_modalities,
        ):
            self._metrics.record_skip()
            logger.debug(
                "[WorkflowCoordinator] Skipping explainability for %s (conf=%.2f modality=%s).",
                request_id, confidence, modality,
            )
            return prediction_result

        # Steps 2–8: Execute explainability pipeline via ExecutionManager
        try:
            xai_result, timings = self._exec_mgr.execute(engine_ctx, prediction_result)
        except Exception as exc:
            logger.error("[WorkflowCoordinator] ExecutionManager raised for %s: %s", request_id, exc)
            if not self._config.fail_silently:
                raise WorkflowCoordinationError(str(exc), request_id=request_id) from exc
            timings = ExplainabilityTimings(request_id=request_id, success=False, modality=modality)
            self._metrics.record(timings)
            return prediction_result

        self._metrics.record(timings)

        # Step 9: Attach ExplainabilityResult to PredictionResult
        t_attach = time.perf_counter()
        enriched = self._attach_to_prediction_result(prediction_result, xai_result, timings)
        timings.result_attachment_ms = (time.perf_counter() - t_attach) * 1000

        logger.info(
            "[WorkflowCoordinator] Workflow complete for %s: status=%s total=%.1fms",
            request_id, xai_result.status, timings.total_explainability_ms,
        )

        # Step 10: Return unified platform response
        return enriched

    # ── Internal: Attach to PredictionResult ─────────────────────────────────

    def _attach_to_prediction_result(
        self,
        prediction_result: Any,
        xai_result: Any,
        timings: "ExplainabilityTimings",
    ) -> Any:
        """
        Attach ExplainabilityResult data to a frozen PredictionResult by rebuilding it.

        Since PredictionResult is a frozen dataclass, we use dataclasses.replace()
        to produce a new instance with updated extension fields.
        """
        try:
            gradcam_dict = {
                "status": xai_result.status,
                "method": xai_result.method,
                "target_layer": xai_result.target_layer,
                "heatmap_path": xai_result.heatmap_path,
                "overlay_path": xai_result.overlay_path,
                "important_regions": list(xai_result.important_regions),
                "clinical_summary": xai_result.clinical_summary,
                "quality_score": xai_result.quality_score,
                "generation_time_ms": xai_result.generation_time_ms,
                "performance_metrics": dict(xai_result.performance_metrics),
                "warnings": list(xai_result.warnings),
                "errors": list(xai_result.errors),
            }

            # Merge extra data (clinical report, recommendations, severity)
            extra_data = dict(xai_result.extra)
            extra_data["explainability_timings"] = timings.to_dict()

            # Updated performance metrics to include explainability timing
            updated_perf = dict(prediction_result.performance_metrics)
            updated_perf["explainability_ms"] = round(timings.total_explainability_ms, 3)
            updated_perf["gradcam_ms"]        = round(timings.gradcam_ms, 3)
            updated_perf["clinical_xai_ms"]   = round(timings.clinical_explanation_ms, 3)

            enriched = replace(
                prediction_result,
                gradcam_result=gradcam_dict,
                performance_metrics=updated_perf,
                extra={**dict(prediction_result.extra), **extra_data},
            )
            return enriched

        except Exception as exc:
            logger.warning(
                "[WorkflowCoordinator] Could not attach ExplainabilityResult: %s", exc
            )
            return prediction_result

    # ── Metrics Access ────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        return self._metrics.snapshot()


__all__ = ["WorkflowCoordinator"]
