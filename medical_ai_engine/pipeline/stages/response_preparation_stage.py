"""
Medical AI Engine — ResponsePreparationStage (Stage 8)

Assembles normalized prediction result dictionary into ctx.prediction_result,
pre-validating structure for downstream Response Builder (Module 14.2.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class ResponsePreparationStage(AbstractPipelineStage):
    """
    Stage 8: Prediction Response Assembly & Structuring.
    """

    def __init__(self) -> None:
        super().__init__("ResponsePreparationStage")

    def validate(self, ctx: EngineContext) -> None:
        if ctx.disease_prediction is None and ctx.modality_prediction is None:
            raise StageValidationException(
                "Neither modality_prediction nor disease_prediction available to assemble response.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Assembling normalized prediction response...", stage=self.name)

        status_str = "SUCCESS" if not ctx.has_errors() else "COMPLETED_WITH_WARNINGS"

        ctx.prediction_result = {
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "status": status_str,
            "modality": (ctx.modality_prediction or {}).get("detected_modality"),
            "disease_prediction": ctx.disease_prediction,
            "confidence": ctx.confidence,
            "clinical_report": ctx.clinical_report,
            "execution_state": ctx.execution_state,
            "performance_metrics": ctx.performance_metrics,
            "warnings_count": len(ctx.warnings),
            "errors_count": len(ctx.errors),
        }

        ctx.set_timestamp("prediction_json_ready")
        ctx.log("Response preparation complete. Prediction JSON ready.", stage=self.name)

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back ResponsePreparationStage...", stage=self.name)
        ctx.prediction_result = None
