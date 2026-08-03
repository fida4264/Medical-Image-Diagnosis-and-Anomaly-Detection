"""
Medical AI Engine — PipelineExecutor

Core orchestrator of the 9-stage prediction execution pipeline.
Thread-safe, handles lifecycle stage dispatching, event publication, rollback in reverse order,
performance metrics recording, and error propagation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.events.engine_events import EngineEventPublisher
from medical_ai_engine.events.stage_events import (
    StageCompletedEvent,
    StageFailedEvent,
    StageRollbackEvent,
    StageStartedEvent,
)
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage
from medical_ai_engine.pipeline.pipeline_exceptions import (
    PipelineExecutionException,
    StageRollbackException,
)
from medical_ai_engine.pipeline.pipeline_stage_factory import PipelineStageFactory
from medical_ai_engine.utils.timer import EngineTimer


class PipelineExecutor:
    """
    Executes ordered pipeline stages on a shared EngineContext.

    Responsibilities:
      - Sequential stage dispatch
      - Passes identical EngineContext reference across stages
      - Graceful failure handling and rollback in reverse order
      - Event publication before and after each stage (EngineEventPublisher)
      - Stage-level and overall pipeline metrics recording
    """

    def __init__(
        self,
        stages: Optional[List[BasePipelineStage]] = None,
        event_publisher: Optional[EngineEventPublisher] = None,
    ) -> None:
        self.stages = stages if stages is not None else PipelineStageFactory.create_default_pipeline()
        self.logger = logging.getLogger("PipelineExecutor")
        self.event_publisher = event_publisher or EngineEventPublisher(logger=self.logger)
        self._lock = threading.RLock()

    def execute(self, ctx: EngineContext) -> EngineContext:
        """
        Execute all registered pipeline stages sequentially on the shared EngineContext.

        Args:
            ctx: Single shared EngineContext state carrier.

        Returns:
            Populated EngineContext upon successful pipeline completion.

        Raises:
            PipelineExecutionException: If any stage process() or validate() fails.
        """

        completed_stages: List[BasePipelineStage] = []
        pipeline_timer = EngineTimer()
        pipeline_timer.__enter__()

        ctx.log(f"Starting PipelineExecutor with {len(self.stages)} stages.", stage="PipelineExecutor")

        for idx, stage in enumerate(self.stages):
            stage_name = stage.name
            ctx.log(f"Entering pipeline stage [{idx+1}/{len(self.stages)}]: {stage_name}", stage=stage_name)

            # Publish StageStartedEvent
            self.event_publisher.publish(
                StageStartedEvent(
                    stage_name=stage_name,
                    request_id=ctx.request_id,
                    session_id=ctx.session_id,
                    engine_state=ctx.execution_state,
                )
            )

            stage_timer = EngineTimer()
            stage_timer.__enter__()

            try:
                stage.validate(ctx)
                stage.process(ctx)
                stage_timer.__exit__(None, None, None)

                completed_stages.append(stage)
                ctx.record_metric(f"{stage_name}_ms", stage_timer.elapsed_ms)

                # Publish StageCompletedEvent
                self.event_publisher.publish(
                    StageCompletedEvent(
                        stage_name=stage_name,
                        duration_ms=stage_timer.elapsed_ms,
                        request_id=ctx.request_id,
                        session_id=ctx.session_id,
                        engine_state=ctx.execution_state,
                    )
                )
                ctx.log(
                    f"Completed stage {stage_name} in {stage_timer.elapsed_ms:.2f}ms",
                    stage=stage_name,
                )

            except Exception as e:
                stage_timer.__exit__(None, None, None)
                err_msg = str(e)
                ctx.log(f"Stage {stage_name} FAILED: {err_msg}", stage=stage_name)
                ctx.add_error(stage=stage_name, error=err_msg)

                # Publish StageFailedEvent
                self.event_publisher.publish(
                    StageFailedEvent(
                        stage_name=stage_name,
                        error_message=err_msg,
                        request_id=ctx.request_id,
                        session_id=ctx.session_id,
                        engine_state=ctx.execution_state,
                    )
                )

                # Execute Rollback in REVERSE order on completed stages
                self._rollback_stages(completed_stages, ctx)

                # Clean up attempted stage + completed stages
                attempted = completed_stages + [stage]
                self._cleanup_stages(attempted, ctx)

                pipeline_timer.__exit__(None, None, None)
                ctx.record_metric("total_pipeline_ms", pipeline_timer.elapsed_ms)

                raise PipelineExecutionException(
                    message=f"Pipeline execution failed at stage '{stage_name}': {err_msg}",
                    stage_name=stage_name,
                    failed_stage_index=idx,
                    cause=e,
                ) from e

        # Clean up all stages on success
        self._cleanup_stages(self.stages, ctx)

        pipeline_timer.__exit__(None, None, None)
        ctx.record_metric("total_pipeline_ms", pipeline_timer.elapsed_ms)
        ctx.log(
            f"PipelineExecutor completed all {len(self.stages)} stages successfully in {pipeline_timer.elapsed_ms:.2f}ms.",
            stage="PipelineExecutor",
        )

        return ctx

    def _rollback_stages(self, completed_stages: List[BasePipelineStage], ctx: EngineContext) -> None:
        """
        Execute rollback() on completed stages in REVERSE order.
        """
        ctx.log("Initiating pipeline rollback on completed stages in reverse order...", stage="PipelineExecutor")
        for stage in reversed(completed_stages):
            stage_name = stage.name
            try:
                ctx.log(f"Rolling back stage: {stage_name}", stage=stage_name)
                stage.rollback(ctx)
                self.event_publisher.publish(
                    StageRollbackEvent(
                        stage_name=stage_name,
                        request_id=ctx.request_id,
                        session_id=ctx.session_id,
                        engine_state=ctx.execution_state,
                    )
                )
            except Exception as rollback_err:
                ctx.log(f"Error rolling back stage {stage_name}: {rollback_err}", stage=stage_name)

    def _cleanup_stages(self, stages: List[BasePipelineStage], ctx: EngineContext) -> None:
        """
        Execute cleanup() on stages in REVERSE order.
        """
        for stage in reversed(stages):
            try:
                stage.cleanup(ctx)
            except Exception as cleanup_err:
                ctx.log(f"Error during stage {stage.name} cleanup: {cleanup_err}", stage=stage.name)
