"""
Medical AI Engine — Engine State & Lifecycle

Defines EngineState enum and EngineLifecycle state machine.
Illegal transitions raise EngineStateException.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, FrozenSet, Optional
import threading

from medical_ai_engine.core.exceptions import EngineStateException


class EngineState(str, Enum):
    """
    Valid engine lifecycle states.

    Transitions:
        CREATED      → INITIALIZED
        INITIALIZED  → RUNNING | FAILED | CANCELLED
        RUNNING      → WAITING | COMPLETED | FAILED | CANCELLED
        WAITING      → RUNNING | FAILED | CANCELLED
        COMPLETED    → (terminal)
        FAILED       → (terminal)
        CANCELLED    → (terminal)
    """

    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        """Returns True if no further transitions are allowed."""
        return self in _TERMINAL_STATES

    def is_active(self) -> bool:
        """Returns True if the engine is currently processing."""
        return self in (EngineState.RUNNING, EngineState.WAITING)

    def can_transition_to(self, target: "EngineState") -> bool:
        """Check whether a transition from self → target is legal."""
        return target in _ALLOWED_TRANSITIONS.get(self, frozenset())


# ── State machine definition ──────────────────────────────────────────────────

_TERMINAL_STATES: FrozenSet[EngineState] = frozenset(
    {EngineState.COMPLETED, EngineState.FAILED, EngineState.CANCELLED}
)

_ALLOWED_TRANSITIONS: Dict[EngineState, FrozenSet[EngineState]] = {
    EngineState.CREATED: frozenset(
        {EngineState.INITIALIZED, EngineState.FAILED, EngineState.CANCELLED}
    ),
    EngineState.INITIALIZED: frozenset(
        {EngineState.RUNNING, EngineState.COMPLETED, EngineState.FAILED, EngineState.CANCELLED}
    ),
    EngineState.RUNNING: frozenset(
        {
            EngineState.WAITING,
            EngineState.COMPLETED,
            EngineState.FAILED,
            EngineState.CANCELLED,
        }
    ),
    EngineState.WAITING: frozenset(
        {EngineState.RUNNING, EngineState.FAILED, EngineState.CANCELLED}
    ),
    EngineState.COMPLETED: frozenset(),
    EngineState.FAILED: frozenset(),
    EngineState.CANCELLED: frozenset(),
}


class EngineLifecycle:
    """
    Thread-safe state machine for engine lifecycle management.

    Enforces valid transitions and records history.
    Raises EngineStateException on illegal transitions.

    Usage:
        lifecycle = EngineLifecycle(request_id="req_001")
        lifecycle.transition(EngineState.INITIALIZED)
        lifecycle.transition(EngineState.RUNNING)
        lifecycle.transition(EngineState.COMPLETED)
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        initial_state: EngineState = EngineState.CREATED,
    ) -> None:
        self._state: EngineState = initial_state
        self._request_id = request_id
        self._lock = threading.RLock()
        self._history: list[dict] = [
            {"state": initial_state.value, "timestamp": _utc_now()}
        ]

    @property
    def state(self) -> EngineState:
        """Current engine state (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def history(self) -> list[dict]:
        """Immutable snapshot of state transition history."""
        with self._lock:
            return list(self._history)

    def transition(self, target: EngineState) -> None:
        """
        Transition to a new state.

        Args:
            target: The desired next state.

        Raises:
            EngineStateException: If the transition is not allowed.
        """
        with self._lock:
            if not self._state.can_transition_to(target):
                allowed = sorted(
                    s.value for s in _ALLOWED_TRANSITIONS.get(self._state, frozenset())
                )
                raise EngineStateException(
                    current_state=self._state.value,
                    attempted_transition=target.value,
                    allowed_transitions=allowed,
                    request_id=self._request_id,
                )
            self._state = target
            self._history.append(
                {"state": target.value, "timestamp": _utc_now()}
            )

    def is_terminal(self) -> bool:
        """Returns True if no further transitions are allowed."""
        return self._state.is_terminal()

    def is_active(self) -> bool:
        """Returns True if engine is RUNNING or WAITING."""
        return self._state.is_active()

    def reset(self) -> None:
        """
        Reset lifecycle to CREATED state.
        Only permitted from terminal states.
        """
        with self._lock:
            if not self._state.is_terminal():
                raise EngineStateException(
                    current_state=self._state.value,
                    attempted_transition=EngineState.CREATED.value,
                    allowed_transitions=[],
                    request_id=self._request_id,
                )
            self._state = EngineState.CREATED
            self._history.append(
                {"state": EngineState.CREATED.value, "timestamp": _utc_now(), "action": "RESET"}
            )

    def __repr__(self) -> str:
        return f"EngineLifecycle(state={self._state.value}, steps={len(self._history)})"


def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
