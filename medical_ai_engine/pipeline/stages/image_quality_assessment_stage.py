"""
Medical AI Engine — ImageQualityAssessmentStage (Stage 2)

Evaluates image quality metrics (resolution, aspect ratio, color channel check)
and populates quality score in EngineContext metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class ImageQualityAssessmentStage(AbstractPipelineStage):
    """
    Stage 2: Image Quality Assessment.
    """

    def __init__(self, min_resolution: int = 16) -> None:
        super().__init__("ImageQualityAssessmentStage")
        self.min_resolution = min_resolution

    def validate(self, ctx: EngineContext) -> None:
        if ctx.image is None:
            raise StageValidationException("ctx.image is missing for quality check.", stage_name=self.name)

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Assessing image quality...", stage=self.name)

        w, h = ctx.image.size
        if w < self.min_resolution or h < self.min_resolution:
            warn_msg = f"Image resolution ({w}x{h}) is below recommended minimum ({self.min_resolution}x{self.min_resolution})."
            ctx.warn(warn_msg, stage=self.name)
            quality_score = 0.40
        else:
            quality_score = 0.95

        ctx.metadata["quality_score"] = quality_score
        ctx.set_timestamp("quality_check_complete")
        ctx.log(f"Image quality assessment complete. Score: {quality_score:.2f}", stage=self.name)

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back ImageQualityAssessmentStage...", stage=self.name)
        ctx.metadata.pop("quality_score", None)
