"""
Medical AI Engine — Pipeline Stage Events

Concrete event classes published during pipeline stage execution:
StageStartedEvent, StageCompletedEvent, StageFailedEvent, StageRollbackEvent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from medical_ai_engine.events.engine_events import BaseEngineEvent


@dataclass
class StageStartedEvent(BaseEngineEvent):
    """Fired immediately before a pipeline stage process() call."""
    stage_name: str = ""

    def __init__(self, stage_name: str, **kwargs) -> None:
        super().__init__(event_name="StageStarted", **kwargs)
        self.stage_name = stage_name
        self.payload["stage_name"] = stage_name


@dataclass
class StageCompletedEvent(BaseEngineEvent):
    """Fired immediately after a pipeline stage completes process() successfully."""
    stage_name: str = ""
    duration_ms: float = 0.0

    def __init__(self, stage_name: str, duration_ms: float = 0.0, **kwargs) -> None:
        super().__init__(event_name="StageCompleted", **kwargs)
        self.stage_name = stage_name
        self.duration_ms = duration_ms
        self.payload["stage_name"] = stage_name
        self.payload["duration_ms"] = duration_ms


@dataclass
class StageFailedEvent(BaseEngineEvent):
    """Fired when a stage process() or validate() raises an exception."""
    stage_name: str = ""
    error_message: str = ""

    def __init__(self, stage_name: str, error_message: str = "", **kwargs) -> None:
        super().__init__(event_name="StageFailed", **kwargs)
        self.stage_name = stage_name
        self.error_message = error_message
        self.payload["stage_name"] = stage_name
        self.payload["error_message"] = error_message


@dataclass
class StageRollbackEvent(BaseEngineEvent):
    """Fired when a stage rollback() is triggered following a pipeline failure."""
    stage_name: str = ""

    def __init__(self, stage_name: str, **kwargs) -> None:
        super().__init__(event_name="StageRollback", **kwargs)
        self.stage_name = stage_name
        self.payload["stage_name"] = stage_name
