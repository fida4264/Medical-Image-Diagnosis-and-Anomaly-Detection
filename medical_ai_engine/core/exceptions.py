"""
Medical AI Engine — Exception Hierarchy

Defines the complete exception hierarchy for the MedicalAIEngine Core.
Every exception carries: timestamp, request_id, engine_state, error_code, message.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


class EngineException(Exception):
    """
    Base exception for all Medical AI Engine errors.

    Attributes:
        message:      Human-readable error description.
        error_code:   Machine-readable error identifier (e.g. 'ENG-001').
        request_id:   Optional request ID that triggered the error.
        engine_state: Optional string representation of engine state at failure.
        timestamp:    UTC timestamp when the exception was raised.
        cause:        Optional chained original exception.
    """

    ERROR_CODE = "ENG-000"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        engine_state: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.ERROR_CODE
        self.request_id = request_id
        self.engine_state = engine_state
        self.timestamp = datetime.now(tz=timezone.utc).isoformat()
        self.cause = cause

    def to_dict(self) -> dict:
        """Serialize exception to a JSON-compatible dictionary."""
        return {
            "exception_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
            "engine_state": self.engine_state,
            "timestamp": self.timestamp,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.engine_state:
            parts.append(f"engine_state={self.engine_state}")
        parts.append(f"timestamp={self.timestamp}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s})"


# ── Initialization ─────────────────────────────────────────────────────────────

class EngineInitializationException(EngineException):
    """
    Raised when the engine fails to initialize.

    Common causes:
        - Invalid configuration file or schema.
        - Missing required directories.
        - Failed dependency checks.
    """

    ERROR_CODE = "ENG-100"

    def __init__(
        self,
        message: str = "Engine initialization failed.",
        **kwargs,
    ) -> None:
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


class ConfigurationException(EngineInitializationException):
    """
    Raised when engine configuration is invalid or cannot be loaded.

    Common causes:
        - YAML parse error.
        - Missing required configuration keys.
        - Invalid configuration values (e.g. negative timeout).
    """

    ERROR_CODE = "ENG-101"

    def __init__(
        self,
        message: str = "Engine configuration is invalid or missing.",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.config_path = config_path
        if config_path:
            message = f"{message} [config_path={config_path}]"
        super().__init__(message, **kwargs)


# ── State Machine ──────────────────────────────────────────────────────────────

class EngineStateException(EngineException):
    """
    Raised when an illegal state transition is attempted.

    The engine enforces a strict lifecycle:
        CREATED → INITIALIZED → RUNNING → COMPLETED / FAILED / CANCELLED

    Any transition not in the allowed map raises this exception.
    """

    ERROR_CODE = "ENG-200"

    def __init__(
        self,
        current_state: str,
        attempted_transition: str,
        allowed_transitions: Optional[list] = None,
        **kwargs,
    ) -> None:
        self.current_state = current_state
        self.attempted_transition = attempted_transition
        self.allowed_transitions = allowed_transitions or []
        message = (
            f"Illegal state transition: {current_state!r} → {attempted_transition!r}. "
            f"Allowed transitions from {current_state!r}: {self.allowed_transitions}"
        )
        super().__init__(
            message,
            error_code=self.ERROR_CODE,
            engine_state=current_state,
            **kwargs,
        )


# ── Execution ──────────────────────────────────────────────────────────────────

class EngineExecutionException(EngineException):
    """
    Raised when an error occurs during engine execution (after initialization).

    Common causes:
        - Pipeline stage failure.
        - Session timeout.
        - Resource exhaustion.
    """

    ERROR_CODE = "ENG-300"

    def __init__(
        self,
        message: str = "Engine execution failed.",
        stage_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.stage_name = stage_name
        if stage_name:
            message = f"{message} [stage={stage_name}]"
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


class ExecutionTimeoutException(EngineExecutionException):
    """
    Raised when an execution session exceeds its configured timeout.
    """

    ERROR_CODE = "ENG-301"

    def __init__(
        self,
        timeout_seconds: float,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        message = f"Execution timed out after {timeout_seconds:.1f}s."
        super().__init__(message, request_id=request_id, **kwargs)


class SessionNotFoundException(EngineExecutionException):
    """
    Raised when a session lookup by ID returns no result.
    """

    ERROR_CODE = "ENG-302"

    def __init__(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session not found: {session_id!r}",
            **kwargs,
        )


# ── Context ────────────────────────────────────────────────────────────────────

class ContextException(EngineException):
    """
    Raised when an EngineContext is invalid, missing required fields,
    or has been mutated illegally.
    """

    ERROR_CODE = "ENG-400"

    def __init__(
        self,
        message: str = "EngineContext is invalid or corrupt.",
        field_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.field_name = field_name
        if field_name:
            message = f"{message} [field={field_name}]"
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


# ── Health & Statistics ────────────────────────────────────────────────────────

class HealthCheckException(EngineException):
    """
    Raised when a health monitor detects an unrecoverable engine issue.
    """

    ERROR_CODE = "ENG-500"

    def __init__(
        self,
        message: str = "Health check failed.",
        check_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.check_name = check_name
        if check_name:
            message = f"{message} [check={check_name}]"
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


# ── Event Bus ─────────────────────────────────────────────────────────────────

class EventPublishException(EngineException):
    """
    Raised when an event cannot be published to the event bus.
    """

    ERROR_CODE = "ENG-600"

    def __init__(
        self,
        event_name: str,
        cause: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        self.event_name = event_name
        super().__init__(
            f"Failed to publish event: {event_name!r}",
            error_code=self.ERROR_CODE,
            cause=cause,
            **kwargs,
        )


# ── Registry ──────────────────────────────────────────────────────────────────

class RegistryException(EngineException):
    """
    Raised when the engine component registry encounters an error.
    (e.g. double-registration, lookup failure)
    """

    ERROR_CODE = "ENG-700"

    def __init__(
        self,
        message: str = "Registry operation failed.",
        component_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.component_name = component_name
        if component_name:
            message = f"{message} [component={component_name}]"
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


# ── Public Error Code Reference ───────────────────────────────────────────────

ERROR_CODES = {
    "ENG-000": "Base engine exception",
    "ENG-100": "Engine initialization failed",
    "ENG-101": "Configuration invalid or missing",
    "ENG-200": "Illegal state transition",
    "ENG-300": "Execution error",
    "ENG-301": "Execution timeout",
    "ENG-302": "Session not found",
    "ENG-400": "EngineContext invalid",
    "ENG-500": "Health check failed",
    "ENG-600": "Event publish failed",
    "ENG-700": "Registry error",
}
