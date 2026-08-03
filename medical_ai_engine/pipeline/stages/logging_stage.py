"""
Medical AI Engine — LoggingStage (Stage 9)

Final stage of pipeline execution. Records execution summary and populates audit logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class LoggingStage(AbstractPipelineStage):
    """
    Stage 9: Audit Trail & Diagnostics Logging.
    """

    def __init__(self) -> None:
        super().__init__("LoggingStage")

    def process(self, ctx: EngineContext) -> None:
        summary = ctx.summary()
        ctx.log(
            f"Pipeline execution completed summary: modality='{summary.get('modality')}', "
            f"diagnosis='{summary.get('predicted_class')}', confidence={summary.get('confidence')}, "
            f"logs={summary.get('log_entries')}, warnings={summary.get('warnings')}, "
            f"errors={summary.get('errors')}",
            stage=self.name,
        )
        ctx.set_timestamp("logged")
