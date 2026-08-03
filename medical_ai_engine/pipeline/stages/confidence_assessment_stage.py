"""
Medical AI Engine — ConfidenceAssessmentStage (Stage 6)

Normalizes disease prediction confidence scores, computes diagnostic reliability,
determines manual review flags, and populates ctx.confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class ConfidenceAssessmentStage(AbstractPipelineStage):
    """
    Stage 6: Confidence & Reliability Assessment.
    """

    def __init__(
        self,
        high_threshold: float = 0.85,
        moderate_threshold: float = 0.70,
    ) -> None:
        super().__init__("ConfidenceAssessmentStage")
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    def validate(self, ctx: EngineContext) -> None:
        if ctx.disease_prediction is None:
            raise StageValidationException(
                "ctx.disease_prediction is missing. AdapterPredictionStage must run prior.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Assessing prediction confidence and reliability...", stage=self.name)

        raw_pred = ctx.disease_prediction or {}
        score = float(raw_pred.get("confidence", 0.0))

        if score >= self.high_threshold:
            reliability = "high"
        elif score >= self.moderate_threshold:
            reliability = "moderate"
        else:
            reliability = "low"

        requires_review = (reliability == "low") or bool(raw_pred.get("requires_manual_review", False))

        ctx.confidence = {
            "score": round(score, 6),
            "reliability": reliability,
            "requires_manual_review": requires_review,
            "severity_grade": raw_pred.get("severity_grade"),
        }

        ctx.set_timestamp("confidence_analysed")
        ctx.log(
            f"Confidence assessment: score={score:.4f}, reliability='{reliability}', "
            f"requires_manual_review={requires_review}",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back ConfidenceAssessmentStage...", stage=self.name)
        ctx.confidence = None
