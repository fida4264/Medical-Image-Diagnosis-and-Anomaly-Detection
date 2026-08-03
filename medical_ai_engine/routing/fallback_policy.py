"""
Medical AI Engine — Fallback Policy

Evaluates modality support and adapter availability.
Implements IFallbackPolicy interface.
"""

from __future__ import annotations

from typing import Optional
from medical_ai_engine.interfaces.routing_policy_interface import IFallbackPolicy
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import (
    RoutingDecision,
    make_model_not_registered,
    make_unsupported_modality,
)


class FallbackPolicy(IFallbackPolicy):
    """
    Handles unsupported modalities and unregistered adapters.
    """

    def __init__(self, config: Optional[RoutingConfiguration] = None) -> None:
        self.config = config or RoutingConfiguration()

    def evaluate_modality(
        self,
        modality: str,
        confidence: float = 0.0,
        request_id: Optional[str] = None,
        routing_time_ms: float = 0.0,
    ) -> Optional[RoutingDecision]:
        """
        Check if modality is supported by the system.

        Returns:
            None if supported.
            RoutingDecision (UNSUPPORTED_MODALITY) if unsupported.
        """
        req_id = request_id or "unknown_req"
        key = modality.lower()

        if not self.config.is_supported(key):
            return make_unsupported_modality(
                request_id=req_id,
                modality=key,
                confidence=confidence,
                supported=self.config.supported_modalities,
                routing_time_ms=routing_time_ms,
            )

        return None

    def evaluate_adapter(
        self,
        modality: str,
        adapter_registered: bool,
        confidence: float = 0.0,
        request_id: Optional[str] = None,
        routing_time_ms: float = 0.0,
    ) -> Optional[RoutingDecision]:
        """
        Check if an adapter is registered in the ModelRegistry for the modality.

        Returns:
            None if adapter is registered.
            RoutingDecision (MODEL_NOT_REGISTERED) if adapter is missing.
        """
        req_id = request_id or "unknown_req"
        key = modality.lower()

        if not adapter_registered:
            return make_model_not_registered(
                request_id=req_id,
                modality=key,
                confidence=confidence,
                routing_time_ms=routing_time_ms,
            )

        return None
