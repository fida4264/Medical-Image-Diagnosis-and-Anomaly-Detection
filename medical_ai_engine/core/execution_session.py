"""
Medical AI Engine — ExecutionSession

Responsible for session tracking, timestamps, duration, request correlation,
execution metadata, thread ownership, and future distributed execution compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Dict, Optional

from medical_ai_engine.core.engine_state import EngineLifecycle, EngineState


class ExecutionSession:
    """
    Tracks and manages a single execution session for a prediction request.

    Captures start/end timestamps, session duration, thread/process ownership,
    request correlation, and holds the EngineLifecycle state machine.

    Thread-safe and designed for future distributed execution compatibility.
    """

    def __init__(
        self,
        request_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id or f"session_{request_id}"
        self.metadata = metadata or {}

        # Initialize session lifecycle in CREATED state
        self.lifecycle = EngineLifecycle(request_id=self.request_id, initial_state=EngineState.CREATED)

        self.start_time_monotonic = time.monotonic()
        self.end_time_monotonic: Optional[float] = None

        self.start_time_utc = datetime.now(tz=timezone.utc).isoformat()
        self.end_time_utc: Optional[str] = None

        self.thread_id = threading.get_ident()
        self.thread_name = threading.current_thread().name
        self.process_id = os.getpid()

        self._lock = threading.Lock()

    @property
    def state(self) -> EngineState:
        """Current session lifecycle state."""
        return self.lifecycle.state

    def duration_seconds(self) -> float:
        """
        Return total elapsed execution time in seconds.
        If session is active, returns time elapsed so far.
        """
        with self._lock:
            if self.end_time_monotonic is not None:
                return round(self.end_time_monotonic - self.start_time_monotonic, 6)
            return round(time.monotonic() - self.start_time_monotonic, 6)

    def duration_ms(self) -> float:
        """Return total elapsed execution time in milliseconds."""
        return round(self.duration_seconds() * 1000.0, 3)

    def transition_state(self, target: EngineState) -> None:
        """
        Transition session state and handle completion timestamping.
        Auto-transitions CREATED -> INITIALIZED if transitioning directly to RUNNING.

        Args:
            target: Next EngineState to transition to.
        """
        with self._lock:
            if self.lifecycle.state == EngineState.CREATED and target == EngineState.RUNNING:
                self.lifecycle.transition(EngineState.INITIALIZED)

            self.lifecycle.transition(target)
            if target.is_terminal():
                self.end_time_monotonic = time.monotonic()
                self.end_time_utc = datetime.now(tz=timezone.utc).isoformat()

    def is_active(self) -> bool:
        """Returns True if session is currently active (RUNNING or WAITING)."""
        return self.lifecycle.is_active()

    def is_complete(self) -> bool:
        """Returns True if session has reached a terminal state."""
        return self.lifecycle.is_terminal()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session details to a dictionary."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "request_id": self.request_id,
                "state": self.state.value,
                "start_time": self.start_time_utc,
                "end_time": self.end_time_utc,
                "duration_ms": self.duration_ms(),
                "process_id": self.process_id,
                "thread_id": self.thread_id,
                "thread_name": self.thread_name,
                "metadata": self.metadata,
                "history": self.lifecycle.history,
            }

    def __repr__(self) -> str:
        return (
            f"ExecutionSession(id={self.session_id!r}, "
            f"request={self.request_id!r}, "
            f"state={self.state.value}, "
            f"duration={self.duration_ms():.1f}ms)"
        )
