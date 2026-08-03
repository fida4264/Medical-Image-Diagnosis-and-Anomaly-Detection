"""
Medical AI Engine — Confidence Policy

Evaluates modality detection confidence against per-modality and global thresholds.
Implements IConfidencePolicy interface.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from medical_ai_engine.interfaces.routing_policy_interface import IConfidencePolicy
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import RoutingDecision, make_low_confidence

if TYPE_CHECKING:
    from medical_ai_engine.routing.routing_decision import RoutingDecision


class ConfidencePolicy(IConfidencePolicy):
    """
    Evaluates prediction confidence against configured thresholds.
    """

    def __init__(self, config: Optional[RoutingConfiguration] = None) -> None:
        self.config = config or RoutingConfiguration()

    def get_threshold(self, modality: str) -> float:
        """Return confidence threshold for modality."""
        return self.config.get_threshold(modality)

    def evaluate(
        self,
        modality: str,
        confidence: float,
        request_id: Optional[str] = None,
        routing_time_ms: float = 0.0,
        top_k: Optional[list] = None,
    ) -> Optional[RoutingDecision]:
        """
        Evaluate confidence score.

        Returns:
            None if confidence passes threshold checks.
            RoutingDecision (LOW_CONFIDENCE) if confidence is insufficient.
        """
        req_id = request_id or "unknown_req"
        key = modality.lower()
        threshold = self.get_threshold(key)

        # Check global floor first
        if confidence < self.config.global_confidence_floor:
            return make_low_confidence(
                request_id=req_id,
                modality=key,
                confidence=confidence,
                threshold=self.config.global_confidence_floor,
                routing_time_ms=routing_time_ms,
                top_k=top_k,
            )

        # Check per-modality threshold
        if confidence < threshold:
            return make_low_confidence(
                request_id=req_id,
                modality=key,
                confidence=confidence,
                threshold=threshold,
                routing_time_ms=routing_time_ms,
                top_k=top_k,
            )

        return None
