"""
Medical AI Engine — State Manager

Utility class for validating and managing state transitions across sessions.
"""

from __future__ import annotations

from typing import List, Optional
import threading

from medical_ai_engine.core.engine_state import EngineState, EngineLifecycle


class EngineStateManager:
    """
    Manages state transition validations, history logging, and active session states.
    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_lifecycles: dict[str, EngineLifecycle] = {}

    def register_lifecycle(self, request_id: str, lifecycle: EngineLifecycle) -> None:
        """Track an active request's lifecycle."""
        with self._lock:
            self._active_lifecycles[request_id] = lifecycle

    def unregister_lifecycle(self, request_id: str) -> Optional[EngineLifecycle]:
        """Remove a completed request's lifecycle."""
        with self._lock:
            return self._active_lifecycles.pop(request_id, None)

    def get_lifecycle(self, request_id: str) -> Optional[EngineLifecycle]:
        """Get the EngineLifecycle for a given request_id."""
        with self._lock:
            return self._active_lifecycles.get(request_id)

    def active_request_count(self) -> int:
        """Return the number of currently tracked active lifecycles."""
        with self._lock:
            return len(self._active_lifecycles)
