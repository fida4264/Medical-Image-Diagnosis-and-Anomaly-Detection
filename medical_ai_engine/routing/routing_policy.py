"""
Medical AI Engine — Routing Policy

Composite policy orchestrating ConfidencePolicy and FallbackPolicy evaluation.
Implements IRoutingPolicy interface.
"""

from __future__ import annotations

from typing import Optional
from medical_ai_engine.interfaces.routing_policy_interface import IRoutingPolicy
from medical_ai_engine.routing.confidence_policy import ConfidencePolicy
from medical_ai_engine.routing.fallback_policy import FallbackPolicy
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import RoutingDecision


class RoutingPolicy(IRoutingPolicy):
    """
    Composite policy applying all routing checks in strict evaluation order:
      1. Fallback: Modality Support Check (UNSUPPORTED_MODALITY)
      2. Confidence Policy Check (LOW_CONFIDENCE)
      3. Fallback: Adapter Registration Check (MODEL_NOT_REGISTERED)
    """

    def __init__(
        self,
        config: Optional[RoutingConfiguration] = None,
        confidence_policy: Optional[ConfidencePolicy] = None,
        fallback_policy: Optional[FallbackPolicy] = None,
    ) -> None:
        self.config = config or RoutingConfiguration()
        self.confidence_policy = confidence_policy or ConfidencePolicy(config=self.config)
        self.fallback_policy = fallback_policy or FallbackPolicy(config=self.config)

    def apply(
        self,
        modality: str,
        confidence: float,
        adapter_registered: bool,
        request_id: Optional[str] = None,
        routing_time_ms: float = 0.0,
        top_k: Optional[list] = None,
    ) -> Optional[RoutingDecision]:
        """
        Apply routing policies in order.

        Returns:
            None if all checks pass.
            RoutingDecision if any policy check fails.
        """
        req_id = request_id or "unknown_req"
        key = modality.lower()

        # 1. Supported modality check
        unsupported_dec = self.fallback_policy.evaluate_modality(
            modality=key,
            confidence=confidence,
            request_id=req_id,
            routing_time_ms=routing_time_ms,
        )
        if unsupported_dec is not None:
            return unsupported_dec

        # 2. Confidence threshold check
        low_conf_dec = self.confidence_policy.evaluate(
            modality=key,
            confidence=confidence,
            request_id=req_id,
            routing_time_ms=routing_time_ms,
            top_k=top_k,
        )
        if low_conf_dec is not None:
            return low_conf_dec

        # 3. Adapter registration check
        no_adapter_dec = self.fallback_policy.evaluate_adapter(
            modality=key,
            adapter_registered=adapter_registered,
            confidence=confidence,
            request_id=req_id,
            routing_time_ms=routing_time_ms,
        )
        if no_adapter_dec is not None:
            return no_adapter_dec

        return None
