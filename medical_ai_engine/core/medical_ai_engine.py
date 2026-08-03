"""
Medical AI Engine Core — MedicalAIEngine

The central orchestration engine core of the Medical AI Platform.
Implements IEngine. Independent of all AI modality models (Modules 6-13).

Responsibilities:
  - Receives prediction requests
  - Creates EngineContext
  - Manages execution lifecycle (CREATED -> INITIALIZED -> RUNNING -> COMPLETED / FAILED)
  - Invokes pipeline stages through interfaces only
  - Manages execution state & sessions
  - Publishes lifecycle events (EngineStarted, EngineInitialized, EngineRunning, etc.)
  - Coordinates logging
  - Tracks operational statistics
  - Returns execution status and populated EngineContext
"""

from __future__ import annotations

import logging
from pathlib import Path

from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Optional, Union

from medical_ai_engine.core.engine_configuration import EngineConfiguration
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.engine_state import EngineLifecycle, EngineState
from medical_ai_engine.core.engine_statistics import EngineStatistics
from medical_ai_engine.core.exceptions import (
    EngineException,
    EngineExecutionException,
    EngineInitializationException,
    EngineStateException,
)
from medical_ai_engine.core.execution_session import ExecutionSession
from medical_ai_engine.events.engine_events import EngineEventPublisher
from medical_ai_engine.events.lifecycle_events import (
    EngineCancelledEvent,
    EngineCompletedEvent,
    EngineFailedEvent,
    EngineInitializedEvent,
    EngineRunningEvent,
    EngineStartedEvent,
)
from medical_ai_engine.interfaces.engine_interface import IEngine
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage
from medical_ai_engine.utils.health_monitor import EngineHealthMonitor
from medical_ai_engine.utils.request_id import generate_request_id, generate_session_id
from medical_ai_engine.utils.state_manager import EngineStateManager
from medical_ai_engine.utils.thread_manager import EngineThreadManager
from medical_ai_engine.utils.timer import EngineTimer


class MedicalAIEngine(IEngine):
    """
    Core Operating System for the Medical AI Platform.

    Provides thread-safe request execution, lifecycle state machine enforcement,
    session management, pipeline stage dispatching, event publication, and metrics.

    Does NOT import or depend on any specific AI model or modality implementation.
    """

    def __init__(
        self,
        config: Optional[EngineConfiguration] = None,
        pipeline_stages: Optional[List[BasePipelineStage]] = None,
    ) -> None:
        self.config = config or EngineConfiguration()
        self._lifecycle = EngineLifecycle()
        self._lock = threading.RLock()

        self._setup_logger()
        self.event_publisher = EngineEventPublisher(logger=self.logger)

        from medical_ai_engine.pipeline.pipeline_executor import PipelineExecutor
        from medical_ai_engine.pipeline.pipeline_stage_factory import PipelineStageFactory

        self.pipeline_stages = pipeline_stages if pipeline_stages is not None else PipelineStageFactory.create_default_pipeline()
        self.executor = PipelineExecutor(stages=self.pipeline_stages, event_publisher=self.event_publisher)

        self.statistics = EngineStatistics()
        self.state_manager = EngineStateManager()
        self.thread_manager = EngineThreadManager(
            max_workers=self.config.max_concurrent_threads,
            thread_prefix=self.config.thread_pool_prefix,
        )
        self.health_monitor = EngineHealthMonitor(engine_name=self.config.engine_name)

        self._active_sessions: Dict[str, ExecutionSession] = {}

        # Publish EngineStarted event
        self.event_publisher.publish(
            EngineStartedEvent(
                engine_state=self._lifecycle.state.value,
                payload={"engine_name": self.config.engine_name, "version": self.config.version},
            )
        )
        self.logger.info(
            f"[{self.config.engine_name}] Created. Status: {self._lifecycle.state.value}"
        )

    def _setup_logger(self) -> None:
        """Configure engine logging."""
        self.logger = logging.getLogger(self.config.engine_name)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )

        if self.config.enable_console_logging:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if self.config.enable_file_logging:
            log_file = Path(self.config.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    @property
    def state(self) -> EngineState:
        """Current engine lifecycle state."""
        return self._lifecycle.state

    def initialize(self) -> None:
        """
        Initialize the engine core resources, configuration, and thread pool.
        Transitions state: CREATED -> INITIALIZED.
        """
        with self._lock:
            if self._lifecycle.state == EngineState.INITIALIZED:
                self.logger.info("Engine is already INITIALIZED.")
                return

            with EngineTimer() as timer:
                try:
                    self.config.validate()
                    self.thread_manager.start()
                    self._lifecycle.transition(EngineState.INITIALIZED)
                except Exception as e:
                    self._lifecycle.transition(EngineState.FAILED)
                    self.logger.error(f"Engine initialization failed: {e}")
                    raise EngineInitializationException(
                        f"Failed to initialize MedicalAIEngine: {e}", cause=e
                    )

            init_ms = timer.elapsed_ms
            self.statistics.record_engine_init(timer.elapsed_seconds)

            self.event_publisher.publish(
                EngineInitializedEvent(
                    init_duration_ms=init_ms,
                    engine_state=self._lifecycle.state.value,
                    payload={"config": self.config.to_dict()},
                )
            )
            self.logger.info(
                f"[{self.config.engine_name}] INITIALIZED in {init_ms:.2f}ms."
            )

    def execute(
        self,
        image_source: Union[str, Path, Any, bytes],
        manual_modality_override: Optional[str] = None,
        request_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EngineContext:
        """
        Execute a prediction request through the complete engine pipeline.

        Args:
            image_source: File path, PIL Image, bytes, or array.
            manual_modality_override: Optional target modality key.
            request_id: Optional UUID string.
            extra_metadata: Optional custom metadata dict.

        Returns:
            EngineContext populated after execution.

        Raises:
            EngineExecutionException: If execution fails.
        """
        if self._lifecycle.state != EngineState.INITIALIZED:
            if self._lifecycle.state == EngineState.CREATED:
                self.initialize()
            else:
                raise EngineStateException(
                    current_state=self._lifecycle.state.value,
                    attempted_transition=EngineState.RUNNING.value,
                    allowed_transitions=["INITIALIZED"],
                )

        req_id = request_id or generate_request_id()
        sess_id = generate_session_id(req_id)

        session = ExecutionSession(
            request_id=req_id,
            session_id=sess_id,
            metadata=extra_metadata or {},
        )

        from PIL import Image as PILModule
        is_pil = isinstance(image_source, PILModule.Image)

        ctx = EngineContext(
            request_id=req_id,
            session_id=sess_id,
            image=image_source if is_pil else None,
            image_source=image_source if isinstance(image_source, (str, Path)) else None,
            image_bytes=image_source if isinstance(image_source, bytes) else None,
            manual_modality_override=manual_modality_override,
            metadata=extra_metadata or {},
        )

        ctx.set_timestamp("request_received")
        ctx.log(f"Request received. Session ID: {sess_id}", stage="CoreEngine")

        with self._lock:
            self._active_sessions[sess_id] = session
            self.state_manager.register_lifecycle(req_id, session.lifecycle)
            self.statistics.session_started()

        session.transition_state(EngineState.RUNNING)
        ctx.execution_state = EngineState.RUNNING.value

        self.event_publisher.publish(
            EngineRunningEvent(
                request_id=req_id,
                session_id=sess_id,
                engine_state=EngineState.RUNNING.value,
            )
        )
        self.logger.info(f"[{req_id}] Execution started (Session: {sess_id}).")

        with EngineTimer() as overall_timer:
            try:
                self._run_pipeline(ctx)
                session.transition_state(EngineState.COMPLETED)
                ctx.execution_state = EngineState.COMPLETED.value
                ctx.set_timestamp("execution_completed")
                ctx.record_metric("total_ms", overall_timer.elapsed_ms)

                self.statistics.session_completed(overall_timer.elapsed_ms)

                self.event_publisher.publish(
                    EngineCompletedEvent(
                        request_id=req_id,
                        session_id=sess_id,
                        duration_ms=overall_timer.elapsed_ms,
                        engine_state=EngineState.COMPLETED.value,
                    )
                )
                self.logger.info(
                    f"[{req_id}] Execution COMPLETED successfully in {overall_timer.elapsed_ms:.2f}ms."
                )

            except Exception as e:
                session.transition_state(EngineState.FAILED)
                ctx.execution_state = EngineState.FAILED.value
                ctx.add_error(stage="ExecutionCore", error=str(e))
                ctx.record_metric("total_ms", overall_timer.elapsed_ms)

                self.statistics.session_failed()

                self.event_publisher.publish(
                    EngineFailedEvent(
                        request_id=req_id,
                        session_id=sess_id,
                        error_message=str(e),
                        engine_state=EngineState.FAILED.value,
                    )
                )
                self.logger.error(f"[{req_id}] Execution FAILED: {e}")
                raise EngineExecutionException(
                    f"Execution failed for request '{req_id}': {e}",
                    request_id=req_id,
                    cause=e,
                )

            finally:
                with self._lock:
                    self._active_sessions.pop(sess_id, None)
                    self.state_manager.unregister_lifecycle(req_id)

        return ctx

    def _run_pipeline(self, ctx: EngineContext) -> None:
        """
        Execute registered pipeline stages sequentially using PipelineExecutor.
        """
        self.executor.stages = self.pipeline_stages
        self.executor.execute(ctx)

    def shutdown(self) -> None:
        """
        Gracefully shutdown the engine, cancel active sessions, stop thread pool.
        """
        with self._lock:
            self.logger.info(f"[{self.config.engine_name}] Shutting down...")

            for session in list(self._active_sessions.values()):
                try:
                    session.transition_state(EngineState.CANCELLED)
                    self.statistics.session_cancelled()
                    self.event_publisher.publish(
                        EngineCancelledEvent(
                            request_id=session.request_id,
                            session_id=session.session_id,
                            reason="Engine shutdown",
                        )
                    )
                except Exception:
                    pass

            self._active_sessions.clear()
            self.thread_manager.shutdown(wait=True)

            if self._lifecycle.state != EngineState.FAILED:
                self._lifecycle.transition(EngineState.COMPLETED)

            self.logger.info(f"[{self.config.engine_name}] Shutdown complete.")

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine metrics snapshot dict."""
        return self.statistics.snapshot()

    def get_health_status(self) -> Dict[str, Any]:
        """Return health monitoring status dict."""
        with self._lock:
            active_cnt = len(self._active_sessions)
            state_val = self._lifecycle.state.value
        return self.health_monitor.check_health(
            current_state=state_val,
            active_sessions_count=active_cnt,
            statistics_snapshot=self.get_statistics(),
        )

    def get_active_sessions(self) -> List[ExecutionSession]:
        """Return snapshot list of active ExecutionSession objects."""
        with self._lock:
            return list(self._active_sessions.values())

    def __repr__(self) -> str:
        return (
            f"MedicalAIEngine(name={self.config.engine_name!r}, "
            f"state={self.state.value}, "
            f"stages={len(self.pipeline_stages)})"
        )
