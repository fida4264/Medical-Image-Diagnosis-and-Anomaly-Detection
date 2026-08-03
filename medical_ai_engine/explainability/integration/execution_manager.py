"""
execution_manager.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Manages the execution lifecycle of one explainability run.

Responsibility
--------------
Given an EngineContext from a completed prediction, the ExecutionManager:
  1. Builds an ExplainabilityContext from the prediction data
  2. Invokes ExplainabilityPipeline (Modules 15.1–15.4)
  3. Invokes ClinicalExplanationEngine (Module 15.5)
  4. Collects per-stage timing metrics into ExplainabilityTimings
  5. Returns an ExplainabilityResult on success, or a FAILED result on error

It does NOT modify EngineContext or PredictionResult (those are immutable or
handled by ExplainabilityIntegrator/WorkflowCoordinator).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.explainability_pipeline import ExplainabilityPipeline
from medical_ai_engine.explainability.explainability_result import ExplainabilityResult

from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import ContextBuildError, ExecutionManagerError
from .integration_metrics import ExplainabilityTimings

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Manages one end-to-end explainability execution lifecycle.

    Produces an ExplainabilityResult and a populated ExplainabilityTimings dict.
    Never raises — errors are captured and returned as a FAILED ExplainabilityResult.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        pipeline: Optional[ExplainabilityPipeline] = None,
    ) -> None:
        self._config = config or IntegrationConfiguration.fast_test()
        self._pipeline = pipeline or ExplainabilityPipeline()

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(
        self,
        engine_ctx: Any,                        # EngineContext (typed loosely to avoid circular imports)
        prediction_result: Any,                 # PredictionResult (frozen dataclass)
    ) -> tuple[ExplainabilityResult, ExplainabilityTimings]:
        """
        Execute the full explainability workflow for one prediction.

        Parameters
        ----------
        engine_ctx:
            EngineContext returned by MedicalAIEngine.execute().
        prediction_result:
            PredictionResult built from engine_ctx.

        Returns
        -------
        (ExplainabilityResult, ExplainabilityTimings)
        """
        request_id = getattr(engine_ctx, "request_id", "unknown")
        session_id  = getattr(engine_ctx, "session_id",  "unknown")
        timings = ExplainabilityTimings(request_id=request_id)
        total_start = time.perf_counter()

        # Stage A: Build ExplainabilityContext
        t0 = time.perf_counter()
        try:
            xai_ctx = self._build_xai_context(engine_ctx, prediction_result, request_id, session_id)
            timings.context_build_ms = (time.perf_counter() - t0) * 1000
            timings.modality = xai_ctx.modality
        except Exception as exc:
            timings.total_explainability_ms = (time.perf_counter() - total_start) * 1000
            timings.success = False
            logger.error("[ExecutionManager] Context build failed for %s: %s", request_id, exc)
            return self._failed_result(request_id, session_id, str(exc)), timings

        # Stage B: Run Explainability Pipeline (15.1–15.4)
        t0 = time.perf_counter()
        result = self._pipeline.execute(xai_ctx)
        pipeline_ms = (time.perf_counter() - t0) * 1000
        # Distribute pipeline time into sub-stage buckets from context metrics
        perf = xai_ctx.performance_metrics
        timings.gradcam_ms     = perf.get("stage3_activation_ms", 0.0)
        timings.heatmap_ms     = perf.get("stage4_heatmap_ms", 0.0)
        timings.overlay_ms     = perf.get("stage5_overlay_ms", 0.0)

        # Stage C: Clinical Explanation Engine (Module 15.5)
        t0 = time.perf_counter()
        if self._config.include_clinical_explanation and result.is_success:
            result = self._attach_clinical_explanation(result, xai_ctx)
        timings.clinical_explanation_ms = (time.perf_counter() - t0) * 1000

        timings.total_explainability_ms = (time.perf_counter() - total_start) * 1000
        timings.success = result.is_success

        logger.info(
            "[ExecutionManager] Explainability for %s: status=%s total=%.1fms",
            request_id, result.status, timings.total_explainability_ms,
        )
        return result, timings

    # ── Internal: Context Builder ─────────────────────────────────────────────

    def _build_xai_context(
        self,
        engine_ctx: Any,
        prediction_result: Any,
        request_id: str,
        session_id: str,
    ) -> ExplainabilityContext:
        """Construct an ExplainabilityContext from EngineContext + PredictionResult."""
        modality = (
            getattr(prediction_result, "modality", None)
            or (getattr(engine_ctx, "modality_prediction", None) or {}).get("detected_modality", "unknown")
        )
        predicted_class = (
            getattr(prediction_result, "predicted_disease", None)
            or (getattr(engine_ctx, "disease_prediction", None) or {}).get("predicted_class", "unknown")
        )
        confidence = float(getattr(prediction_result, "confidence", 0.0))

        # Retrieve image from EngineContext if available
        image = getattr(engine_ctx, "image", None)
        if image is None:
            # Try to reconstruct from bytes
            raw_bytes = getattr(engine_ctx, "image_bytes", None)
            if raw_bytes:
                try:
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception:
                    pass

        return ExplainabilityContext(
            request_id=request_id,
            session_id=session_id,
            image=image,
            modality=modality,
            predicted_class=predicted_class,
            confidence=confidence,
            cam_method_key=self._config.cam_method,
        )

    # ── Internal: Clinical Explanation Attach ────────────────────────────────

    def _attach_clinical_explanation(
        self,
        result: ExplainabilityResult,
        xai_ctx: ExplainabilityContext,
    ) -> ExplainabilityResult:
        """
        Run Module 15.5 ClinicalExplanationEngine and return an updated
        ExplainabilityResult with clinical_summary populated.

        Since ExplainabilityResult is frozen, we rebuild it with the new field.
        """
        try:
            from medical_ai_engine.explainability.clinical import (
                ClinicalContext,
                ClinicalExplanationEngine,
            )
            clinical_ctx = ClinicalContext(
                request_id=xai_ctx.request_id,
                session_id=xai_ctx.session_id,
                modality=xai_ctx.modality,
                disease_label=xai_ctx.predicted_class,
                confidence_score=xai_ctx.confidence,
                important_regions=xai_ctx.important_regions,
            )
            clinical_engine = ClinicalExplanationEngine()
            report = clinical_engine.explain(clinical_ctx)
            clinical_text = report.get("clinical_impression", "")

            # Rebuild frozen result with clinical summary
            result = ExplainabilityResult(
                request_id=result.request_id,
                session_id=result.session_id,
                timestamp=result.timestamp,
                method=result.method,
                modality=result.modality,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                target_layer=result.target_layer,
                heatmap_path=result.heatmap_path,
                overlay_path=result.overlay_path,
                activation_map_path=result.activation_map_path,
                important_regions=result.important_regions,
                clinical_summary=clinical_text,
                quality_score=result.quality_score,
                generation_time_ms=result.generation_time_ms,
                status=result.status,
                performance_metrics={
                    **result.performance_metrics,
                    **{f"clinical_{k}": v for k, v in clinical_ctx.stage_timings_ms.items()},
                },
                warnings=result.warnings,
                errors=result.errors,
                extra={
                    **result.extra,
                    "clinical_report": report,
                    "severity_label": clinical_ctx.severity_label,
                    "recommendations": clinical_ctx.recommendations,
                },
            )
        except Exception as exc:
            logger.warning(
                "[ExecutionManager] Clinical explanation failed for %s: %s",
                xai_ctx.request_id, exc,
            )
        return result

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _failed_result(request_id: str, session_id: str, error: str) -> ExplainabilityResult:
        return ExplainabilityResult(
            request_id=request_id,
            session_id=session_id,
            status="FAILED",
            clinical_summary=f"Explainability execution failed: {error}",
            errors=[{"stage": "ExecutionManager", "error": error}],
        )


__all__ = ["ExecutionManager"]
