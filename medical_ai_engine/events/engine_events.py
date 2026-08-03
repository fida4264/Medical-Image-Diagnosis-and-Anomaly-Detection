"""
Medical AI Engine — Base Engine Event Model

Base event class and event publisher implementation.
Publishes lifecycle events without managing subscribers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import threading

from medical_ai_engine.interfaces.event_publisher import IEventPublisher


@dataclass
class BaseEngineEvent:
    """
    Base class for all engine lifecycle events.
    Carries event_id, event_name, timestamp, request_id, and payload.
    """

    event_name: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    engine_state: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "event_type": self.__class__.__name__,
            "event_name": self.event_name,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "engine_state": self.engine_state,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EngineEventPublisher(IEventPublisher):
    """
    Thread-safe event publisher for MedicalAIEngine.
    Publishes events to loggers/event streams. Does NOT implement subscribers.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("MedicalAIEngine.EventPublisher")
        self._published_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def publish(self, event: BaseEngineEvent) -> None:
        """
        Publish an engine lifecycle event.

        Args:
            event: BaseEngineEvent or subclass instance.
        """
        event_dict = event.to_dict()
        with self._lock:
            self._published_history.append(event_dict)

        req_str = f" [request_id={event.request_id}]" if event.request_id else ""
        state_str = f" [state={event.engine_state}]" if event.engine_state else ""
        self.logger.info(f"EVENT: {event.event_name}{req_str}{state_str}")

    def publish_dict(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event given its name and dictionary payload.
        """
        event = BaseEngineEvent(
            event_name=event_name,
            request_id=payload.get("request_id"),
            session_id=payload.get("session_id"),
            engine_state=payload.get("engine_state"),
            payload=payload,
        )
        self.publish(event)

    def get_published_events(self) -> List[Dict[str, Any]]:
        """Return list of published events history snapshot."""
        with self._lock:
            return list(self._published_history)
