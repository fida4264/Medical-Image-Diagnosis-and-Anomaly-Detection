"""
Medical AI Engine — ModalityClassificationStage (Stage 3)

Invokes Module 6 Modality Classifier (or manual override) via ModalityRouter
to detect the image modality and populate ctx.modality_prediction.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage
from medical_ai_engine.routing.modality_router import ModalityRouter

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class ModalityClassificationStage(AbstractPipelineStage):
    """
    Stage 3: Modality Detection / Classification.
    """

    def __init__(self, router: Optional[ModalityRouter] = None) -> None:
        super().__init__("ModalityClassificationStage")
        self._router = router

    @property
    def router(self) -> ModalityRouter:
        if self._router is None:
            self._router = ModalityRouter()
        return self._router

    def validate(self, ctx: EngineContext) -> None:
        if ctx.image is None and not ctx.manual_modality_override:
            raise StageValidationException(
                "Image or manual_modality_override required for modality classification.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Executing modality classification stage...", stage=self.name)
        decision = self.router.route(ctx)
        ctx.log(
            f"Modality classification completed: outcome={decision.outcome.value}, modality='{decision.detected_modality}'",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back ModalityClassificationStage...", stage=self.name)
        ctx.modality_prediction = None
