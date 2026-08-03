"""
Medical AI Engine — Router Interface

Abstract contract for ModalityRouter implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext
    from medical_ai_engine.routing.routing_decision import RoutingDecision


class IModalityRouter(ABC):
    """
    Abstract interface for the Unified Modality Router.

    The Router receives an EngineContext, invokes the Modality Classifier (Module 6),
    applies confidence and fallback policies, queries the ModelRegistry, and returns
    a RoutingDecision without performing any disease prediction.

    No direct disease model imports are permitted in implementations.
    """

    @abstractmethod
    def route(self, ctx: "EngineContext") -> "RoutingDecision":
        """
        Execute the routing decision for the given EngineContext.

        Steps (internal):
            1. Check manual_modality_override → bypass Module 6 if set
            2. Invoke Module 6 ModalityPredictor
            3. Apply ConfidencePolicy → reject if below threshold
            4. Apply FallbackPolicy → reject if modality unsupported
            5. Query ModelRegistry → verify adapter is registered
            6. Return RoutingDecision with ROUTE_TO_MODEL outcome

        Args:
            ctx: EngineContext carrying image + request state.

        Returns:
            RoutingDecision containing outcome, confidence, adapter_key,
            recommended_action, reason, and timestamps.
        """
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """
        Return health status of the router and its dependencies.

        Returns:
            Dict with keys: status, modality_classifier_loaded, registry_available,
            supported_modalities, confidence_thresholds.
        """
        pass
