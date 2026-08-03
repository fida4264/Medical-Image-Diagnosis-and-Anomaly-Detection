"""
Medical AI Engine — RoutingStage (Stage 4)

Evaluates routing outcome, verifies target adapter registration in ModelRegistry,
and confirms destination for inference execution.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import PipelineExecutionException, StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.routing.routing_decision import RoutingOutcome

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class RoutingStage(AbstractPipelineStage):
    """
    Stage 4: Modality Routing & Adapter Lookup.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        super().__init__("RoutingStage")
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = ModelRegistry.get_instance()
        return self._registry

    def validate(self, ctx: EngineContext) -> None:
        if ctx.modality_prediction is None:
            raise StageValidationException(
                "ctx.modality_prediction is missing. ModalityClassificationStage must run prior to RoutingStage.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Executing routing stage evaluation...", stage=self.name)

        pred_info = ctx.modality_prediction
        outcome_str = pred_info.get("routing_outcome", RoutingOutcome.ROUTING_FAILED.value)
        adapter_key = pred_info.get("adapter_key")

        if outcome_str != RoutingOutcome.ROUTE_TO_MODEL.value:
            reason = pred_info.get("reason", "Routing policy rejected target modality.")
            ctx.log(f"Routing stage rejected: outcome={outcome_str}, reason={reason}", stage=self.name)
            raise PipelineExecutionException(
                f"Routing failed ({outcome_str}): {reason}",
                stage_name=self.name,
            )

        if not adapter_key or not self.registry.is_registered(adapter_key):
            raise PipelineExecutionException(
                f"No adapter registered in ModelRegistry for modality '{adapter_key}'.",
                stage_name=self.name,
            )

        ctx.metadata["selected_adapter"] = adapter_key
        ctx.set_timestamp("adapter_selected")
        ctx.log(
            f"Routing stage verified adapter '{adapter_key}' for prediction.",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back RoutingStage...", stage=self.name)
        ctx.metadata.pop("selected_adapter", None)
