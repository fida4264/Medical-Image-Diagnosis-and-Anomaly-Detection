"""
Medical AI Engine — Concrete Lifecycle Events

Specific event classes published during engine state transitions.
EngineStarted, EngineInitialized, EngineRunning, EngineCompleted, EngineFailed, EngineCancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from medical_ai_engine.events.engine_events import BaseEngineEvent


@dataclass
class EngineStartedEvent(BaseEngineEvent):
    """Fired when the engine process starts up."""
    def __init__(self, **kwargs) -> None:
        super().__init__(event_name="EngineStarted", **kwargs)


@dataclass
class EngineInitializedEvent(BaseEngineEvent):
    """Fired when engine configuration and resources are fully initialized."""
    init_duration_ms: float = 0.0

    def __init__(self, init_duration_ms: float = 0.0, **kwargs) -> None:
        super().__init__(event_name="EngineInitialized", **kwargs)
        self.init_duration_ms = init_duration_ms
        self.payload["init_duration_ms"] = init_duration_ms


@dataclass
class EngineRunningEvent(BaseEngineEvent):
    """Fired when an engine execution session starts processing a request."""
    def __init__(self, **kwargs) -> None:
        super().__init__(event_name="EngineRunning", **kwargs)


@dataclass
class EngineCompletedEvent(BaseEngineEvent):
    """Fired when an engine execution session completes successfully."""
    duration_ms: float = 0.0

    def __init__(self, duration_ms: float = 0.0, **kwargs) -> None:
        super().__init__(event_name="EngineCompleted", **kwargs)
        self.duration_ms = duration_ms
        self.payload["duration_ms"] = duration_ms


@dataclass
class EngineFailedEvent(BaseEngineEvent):
    """Fired when an engine execution session fails with an exception."""
    error_message: str = ""
    error_code: str = "ENG-000"

    def __init__(self, error_message: str = "", error_code: str = "ENG-000", **kwargs) -> None:
        super().__init__(event_name="EngineFailed", **kwargs)
        self.error_message = error_message
        self.error_code = error_code
        self.payload["error_message"] = error_message
        self.payload["error_code"] = error_code


@dataclass
class EngineCancelledEvent(BaseEngineEvent):
    """Fired when an engine execution session is cancelled by user or timeout."""
    reason: str = "User requested cancellation"

    def __init__(self, reason: str = "User requested cancellation", **kwargs) -> None:
        super().__init__(event_name="EngineCancelled", **kwargs)
        self.reason = reason
        self.payload["reason"] = reason
