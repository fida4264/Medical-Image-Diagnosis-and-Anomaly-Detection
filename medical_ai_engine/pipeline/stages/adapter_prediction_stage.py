"""
Medical AI Engine — AdapterPredictionStage (Stage 5)

Retrieves target IPredictorAdapter from ModelRegistry and executes disease prediction on ctx.image.
Populates ctx.disease_prediction.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import PipelineExecutionException, StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage
from medical_ai_engine.registry.model_registry import ModelRegistry

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class AdapterPredictionStage(AbstractPipelineStage):
    """
    Stage 5: Disease Adapter Prediction Execution.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        super().__init__("AdapterPredictionStage")
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = ModelRegistry.get_instance()
        return self._registry

    def validate(self, ctx: EngineContext) -> None:
        adapter_key = ctx.metadata.get("selected_adapter") or (
            ctx.modality_prediction or {}
        ).get("adapter_key")

        if not adapter_key:
            raise StageValidationException(
                "No selected_adapter found in context for prediction.",
                stage_name=self.name,
            )

        if ctx.image is None:
            raise StageValidationException("ctx.image is missing.", stage_name=self.name)

    def process(self, ctx: EngineContext) -> None:
        adapter_key = ctx.metadata.get("selected_adapter") or (
            ctx.modality_prediction or {}
        ).get("adapter_key")
        assert adapter_key is not None

        ctx.log(f"Executing prediction with adapter '{adapter_key}'...", stage=self.name)

        try:
            adapter = self.registry.get_adapter(adapter_key)
            result = adapter.predict(ctx.image)
        except Exception as e:
            raise PipelineExecutionException(
                f"Adapter '{adapter_key}' prediction failed: {e}",
                stage_name=self.name,
                cause=e,
            )

        ctx.disease_prediction = result
        ctx.set_timestamp("disease_predicted")
        ctx.log(
            f"Disease prediction complete: predicted_class='{result.get('predicted_class')}', "
            f"confidence={result.get('confidence')}",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back AdapterPredictionStage...", stage=self.name)
        ctx.disease_prediction = None
