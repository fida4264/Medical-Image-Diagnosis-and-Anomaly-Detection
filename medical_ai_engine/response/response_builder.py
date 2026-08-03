"""
Medical AI Engine — Response Builder

Orchestrates the construction of the final immutable PredictionResult from
a completed EngineContext.

Responsibilities:
  1. Read all relevant fields from EngineContext
  2. Invoke ClinicalReportAggregator to produce a unified clinical report
  3. Assemble model metadata and performance metrics
  4. Compute final PredictionStatus
  5. Construct the immutable PredictionResult (frozen dataclass)
  6. Run ResponseValidator to confirm result integrity
  7. Support error path via build_error_response()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.interfaces.response_builder_interface import IResponseBuilder
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_schema import (
    PredictionStatus,
    PredictionReliability,
    RESPONSE_SCHEMA_VERSION,
)
from medical_ai_engine.response.response_configuration import ResponseConfiguration
from medical_ai_engine.response.response_exceptions import ResponseBuildException
from medical_ai_engine.response.clinical_report_aggregator import ClinicalReportAggregator
from medical_ai_engine.response.response_validator import ResponseValidator

logger = logging.getLogger("ResponseBuilder")


class ResponseBuilder(IResponseBuilder):
    """
    Reads a completed EngineContext and constructs an immutable PredictionResult.

    Thread-safe: a single ResponseBuilder instance may be shared across threads
    because it holds no request-level mutable state.
    """

    def __init__(self, config: Optional[ResponseConfiguration] = None) -> None:
        self.config = config or ResponseConfiguration.default()
        self._aggregator = ClinicalReportAggregator(self.config)
        self._validator = ResponseValidator(self.config)

    # ── Primary Build Path ────────────────────────────────────────────────────

    def build(self, ctx: EngineContext) -> PredictionResult:
        """
        Build an immutable PredictionResult from a fully completed EngineContext.

        Args:
            ctx: EngineContext after successful pipeline execution.

        Returns:
            Frozen PredictionResult.

        Raises:
            ResponseBuildException: If required context fields are absent.
        """
        logger.debug("[ResponseBuilder] Building PredictionResult for request_id=%s", ctx.request_id)

        # Safely extract data with fallbacks
        disease_pred = ctx.disease_prediction or {}
        confidence_data = ctx.confidence or {}
        modality_data = ctx.modality_prediction or {}

        modality = modality_data.get("detected_modality", ctx.manual_modality_override or "unknown")
        predicted_disease = disease_pred.get("predicted_class", "unknown")
        confidence_score = float(disease_pred.get("confidence", 0.0))
        reliability_str = confidence_data.get("reliability", "unknown")
        requires_review = bool(confidence_data.get("requires_manual_review", False))
        severity_grade = confidence_data.get("severity_grade")
        top_k = disease_pred.get("top_k_predictions", [])

        # Clinical report aggregation
        clinical_report = self._aggregator.aggregate(
            modality=modality,
            disease_prediction=disease_pred,
            confidence_data=confidence_data,
            existing_report=ctx.clinical_report,
        )

        # Reliability enum
        reliability = self._to_reliability(reliability_str)

        # Final status determination
        status = self._determine_status(
            ctx=ctx,
            modality=modality,
            confidence_data=confidence_data,
        )

        # Performance metrics
        perf_metrics: Dict[str, float] = {}
        if self.config.include_performance_metrics:
            perf_metrics = dict(ctx.performance_metrics)

        # Model metadata
        model_meta: Dict[str, Any] = {}
        if self.config.include_model_metadata:
            model_meta = self._extract_model_metadata(disease_pred, ctx)

        # Routing decision
        routing_decision: Dict[str, Any] = {}
        if self.config.include_routing_decision:
            routing_decision = self._extract_routing_decision(modality_data, ctx)

        # Extension hooks
        gradcam = ctx.gradcam_result if self.config.include_gradcam_if_present else None
        anomaly = ctx.anomaly_result if self.config.include_anomaly_if_present else None

        result = PredictionResult(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            status=status,
            modality=modality,
            predicted_disease=predicted_disease,
            confidence=confidence_score,
            prediction_reliability=reliability,
            requires_manual_review=requires_review,
            top_predictions=top_k,
            severity_grade=severity_grade,
            clinical_report=clinical_report,
            performance_metrics=perf_metrics,
            model_metadata=model_meta,
            routing_decision=routing_decision,
            warnings=list(ctx.warnings),
            errors=list(ctx.errors),
            gradcam_result=gradcam,
            anomaly_result=anomaly,
            extra=dict(ctx.future_extensions),
        )

        self._validator.validate(result)
        logger.info("[ResponseBuilder] PredictionResult built successfully: %r", result)
        return result

    # ── Error Response Path ───────────────────────────────────────────────────

    def build_error_response(
        self,
        ctx: EngineContext,
        error_message: str,
    ) -> PredictionResult:
        """
        Build a FAILED PredictionResult when the pipeline execution failed.

        Args:
            ctx: Partially populated EngineContext.
            error_message: Description of the failure.

        Returns:
            Frozen PredictionResult with FAILED status.
        """
        logger.warning("[ResponseBuilder] Building error response: %s", error_message)

        errors_list = list(ctx.errors) + [
            {"stage": "ResponseBuilder", "error": error_message, "timestamp": "N/A"}
        ]

        return PredictionResult(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            status=PredictionStatus.FAILED,
            modality="unknown",
            predicted_disease="unknown",
            confidence=0.0,
            prediction_reliability=PredictionReliability.UNKNOWN,
            requires_manual_review=True,
            clinical_report={
                "report_type": "Error Report",
                "primary_diagnosis": "N/A — pipeline failure",
                "reliability": "unknown",
                "requires_manual_review": True,
                "differential_diagnoses": [],
                "clinical_note": f"Pipeline execution failed: {error_message}",
                "recommendations": ["Investigate pipeline failure before re-submission."],
            },
            performance_metrics=dict(ctx.performance_metrics),
            warnings=list(ctx.warnings),
            errors=errors_list,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _determine_status(
        self,
        ctx: EngineContext,
        modality: str,
        confidence_data: Dict[str, Any],
    ) -> PredictionStatus:
        if ctx.has_errors():
            return PredictionStatus.COMPLETED_WITH_WARNINGS
        if ctx.has_warnings():
            return PredictionStatus.COMPLETED_WITH_WARNINGS
        return PredictionStatus.SUCCESS

    def _to_reliability(self, reliability_str: str) -> PredictionReliability:
        """Map string reliability to enum."""
        try:
            return PredictionReliability(reliability_str.lower())
        except (ValueError, AttributeError):
            return PredictionReliability.UNKNOWN

    def _extract_model_metadata(
        self,
        disease_pred: Dict[str, Any],
        ctx: EngineContext,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        for key in ("model_name", "model_version", "inference_backend", "inference_time_ms"):
            if key in disease_pred:
                meta[key] = disease_pred[key]
        # Pull selected_adapter from metadata
        adapter_name = ctx.metadata.get("selected_adapter")
        if adapter_name:
            meta["adapter"] = adapter_name
        return meta

    def _extract_routing_decision(
        self,
        modality_data: Dict[str, Any],
        ctx: EngineContext,
    ) -> Dict[str, Any]:
        return {
            "detected_modality": modality_data.get("detected_modality", "unknown"),
            "modality_confidence": modality_data.get("modality_confidence", 0.0),
            "routing_mode": modality_data.get("routing_mode", "automatic"),
            "routing_time_ms": modality_data.get("routing_time_ms", 0.0),
        }
