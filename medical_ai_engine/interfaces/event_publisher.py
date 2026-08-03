"""
Medical AI Engine — EventPublisher Interface

Abstract contract for lifecycle event publication.
Does NOT implement subscribers — publication only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.events.engine_events import BaseEngineEvent


class IEventPublisher(ABC):
    """
    Abstract interface for publishing engine lifecycle events.
    Decouples event creation from event publishing mechanism.
    """

    @abstractmethod
    def publish(self, event: BaseEngineEvent) -> None:
        """
        Publish an engine lifecycle event.

        Args:
            event: Instance of BaseEngineEvent or subclass.
        """
        pass

    @abstractmethod
    def publish_dict(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event by name and dictionary payload.

        Args:
            event_name: Name identifier of the event.
            payload: Data dictionary associated with the event.
        """
        pass
