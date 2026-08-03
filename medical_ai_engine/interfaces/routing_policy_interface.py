"""
Medical AI Engine — Routing Policy Interface

Abstract contract for policy components used by the Modality Router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.routing.routing_decision import RoutingDecision


class IConfidencePolicy(ABC):
    """
    Abstract contract for confidence evaluation policies.

    Determines whether a modality prediction's confidence score is
    sufficient to proceed with routing.
    """

    @abstractmethod
    def evaluate(
        self,
        modality: str,
        confidence: float,
        request_id: Optional[str] = None,
    ) -> "RoutingDecision | None":
        """
        Evaluate the confidence score for a detected modality.

        Returns:
            None if confidence is acceptable (routing may proceed).
            RoutingDecision with LOW_CONFIDENCE outcome if rejected.
        """
        pass

    @abstractmethod
    def get_threshold(self, modality: str) -> float:
        """Return the configured confidence threshold for a modality."""
        pass


class IFallbackPolicy(ABC):
    """
    Abstract contract for fallback/rejection policy components.

    Determines whether a routing request should be rejected
    due to unsupported modality or unavailable adapter.
    """

    @abstractmethod
    def evaluate_modality(
        self,
        modality: str,
        request_id: Optional[str] = None,
    ) -> "RoutingDecision | None":
        """
        Check if the detected modality is supported for routing.

        Returns:
            None if supported (routing may proceed).
            RoutingDecision with UNSUPPORTED_MODALITY outcome if not.
        """
        pass

    @abstractmethod
    def evaluate_adapter(
        self,
        modality: str,
        registry_healthy: bool,
        request_id: Optional[str] = None,
    ) -> "RoutingDecision | None":
        """
        Check if a registry adapter is available for a modality.

        Returns:
            None if adapter available (routing may proceed).
            RoutingDecision with MODEL_NOT_REGISTERED outcome if not.
        """
        pass


class IRoutingPolicy(ABC):
    """
    Composite routing policy orchestrating confidence + fallback checks.
    """

    @abstractmethod
    def apply(
        self,
        modality: str,
        confidence: float,
        adapter_registered: bool,
        request_id: Optional[str] = None,
    ) -> "RoutingDecision | None":
        """
        Apply all routing policies in order.

        Returns:
            None if all checks pass (routing may proceed to ROUTE_TO_MODEL).
            RoutingDecision with rejection outcome if any policy rejects.
        """
        pass
