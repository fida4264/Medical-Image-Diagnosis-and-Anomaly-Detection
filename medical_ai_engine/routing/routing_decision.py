"""
Medical AI Engine — RoutingDecision

Immutable data structure representing the outcome of a routing operation.
Produced by ModalityRouter and consumed by the prediction pipeline stage.

Outcome values:
  ROUTE_TO_MODEL        — Confidence accepted; adapter found; ready to predict
  LOW_CONFIDENCE        — Module 6 confidence below per-modality threshold
  UNSUPPORTED_MODALITY  — Detected modality not in supported set
  MODEL_NOT_REGISTERED  — Adapter class not registered in ModelRegistry
  ROUTING_FAILED        — Unexpected routing error (catch-all)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingOutcome(str, Enum):
    """Enumeration of all possible routing decision outcomes."""

    ROUTE_TO_MODEL = "ROUTE_TO_MODEL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    MODEL_NOT_REGISTERED = "MODEL_NOT_REGISTERED"
    ROUTING_FAILED = "ROUTING_FAILED"

    @property
    def is_success(self) -> bool:
        """True only for ROUTE_TO_MODEL."""
        return self == RoutingOutcome.ROUTE_TO_MODEL

    @property
    def is_failure(self) -> bool:
        """True for all non-success outcomes."""
        return not self.is_success


@dataclass
class RoutingDecision:
    """
    Immutable routing outcome produced by ModalityRouter.

    Fields:
        outcome           — RoutingOutcome enum value
        request_id        — UUID of the originating request
        detected_modality — Modality string from Module 6 (or override)
        confidence        — Module 6 confidence score (0.0–1.0)
        adapter_key       — Registry key of the selected adapter (if success)
        reason            — Human-readable explanation of the decision
        recommended_action— Short action string for downstream consumers
        routing_mode      — "automatic" | "manual_override"
        routing_time_ms   — Wall-clock time spent routing
        timestamp         — UTC ISO 8601 timestamp at decision creation
        threshold_applied — Confidence threshold that was evaluated
        top_k_modalities  — Top-K modality candidates from Module 6
        warnings          — Non-fatal warnings accumulated during routing
        extra             — Open namespace for future extension data
    """

    # Core outcome fields
    outcome: RoutingOutcome
    request_id: str
    detected_modality: str
    confidence: float

    # Routing metadata
    reason: str
    recommended_action: str
    routing_mode: str = "automatic"
    routing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    # Adapter reference (only populated on ROUTE_TO_MODEL)
    adapter_key: Optional[str] = None

    # Policy evaluation results
    threshold_applied: float = 0.0
    confidence_delta: float = 0.0   # confidence - threshold (positive = above)

    # Module 6 top-K candidates
    top_k_modalities: List[Dict[str, Any]] = field(default_factory=list)

    # Warnings (e.g. confidence is valid but close to threshold)
    warnings: List[str] = field(default_factory=list)

    # Open extension namespace
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_success(self) -> bool:
        """True if outcome is ROUTE_TO_MODEL and adapter_key is set."""
        return self.outcome.is_success and self.adapter_key is not None

    @property
    def is_low_confidence(self) -> bool:
        return self.outcome == RoutingOutcome.LOW_CONFIDENCE

    @property
    def is_unsupported(self) -> bool:
        return self.outcome == RoutingOutcome.UNSUPPORTED_MODALITY

    @property
    def is_not_registered(self) -> bool:
        return self.outcome == RoutingOutcome.MODEL_NOT_REGISTERED

    @property
    def is_failed(self) -> bool:
        return self.outcome == RoutingOutcome.ROUTING_FAILED

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "outcome": self.outcome.value,
            "request_id": self.request_id,
            "detected_modality": self.detected_modality,
            "confidence": round(self.confidence, 6),
            "adapter_key": self.adapter_key,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "routing_mode": self.routing_mode,
            "routing_time_ms": round(self.routing_time_ms, 3),
            "timestamp": self.timestamp,
            "threshold_applied": round(self.threshold_applied, 6),
            "confidence_delta": round(self.confidence_delta, 6),
            "top_k_modalities": self.top_k_modalities,
            "warnings": self.warnings,
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        return (
            f"RoutingDecision("
            f"outcome={self.outcome.value!r}, "
            f"modality={self.detected_modality!r}, "
            f"confidence={self.confidence:.3f}, "
            f"adapter_key={self.adapter_key!r})"
        )


# ── Factory Helpers ───────────────────────────────────────────────────────────

def make_route_to_model(
    request_id: str,
    modality: str,
    confidence: float,
    adapter_key: str,
    threshold: float,
    routing_mode: str = "automatic",
    routing_time_ms: float = 0.0,
    top_k: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
) -> RoutingDecision:
    """Factory: produce a successful ROUTE_TO_MODEL decision."""
    return RoutingDecision(
        outcome=RoutingOutcome.ROUTE_TO_MODEL,
        request_id=request_id,
        detected_modality=modality,
        confidence=confidence,
        adapter_key=adapter_key,
        reason=f"Confidence {confidence:.3f} ≥ threshold {threshold:.3f} for modality '{modality}'.",
        recommended_action="proceed_to_inference",
        routing_mode=routing_mode,
        routing_time_ms=routing_time_ms,
        threshold_applied=threshold,
        confidence_delta=round(confidence - threshold, 6),
        top_k_modalities=top_k or [],
        warnings=warnings or [],
    )


def make_low_confidence(
    request_id: str,
    modality: str,
    confidence: float,
    threshold: float,
    routing_time_ms: float = 0.0,
    top_k: Optional[List[Dict[str, Any]]] = None,
) -> RoutingDecision:
    """Factory: produce a LOW_CONFIDENCE rejection decision."""
    return RoutingDecision(
        outcome=RoutingOutcome.LOW_CONFIDENCE,
        request_id=request_id,
        detected_modality=modality,
        confidence=confidence,
        adapter_key=None,
        reason=(
            f"Modality '{modality}' confidence {confidence:.3f} is below "
            f"required threshold {threshold:.3f}. Manual review required."
        ),
        recommended_action="request_manual_review_or_rescan",
        routing_time_ms=routing_time_ms,
        threshold_applied=threshold,
        confidence_delta=round(confidence - threshold, 6),
        top_k_modalities=top_k or [],
    )


def make_unsupported_modality(
    request_id: str,
    modality: str,
    confidence: float,
    supported: List[str],
    routing_time_ms: float = 0.0,
) -> RoutingDecision:
    """Factory: produce an UNSUPPORTED_MODALITY rejection decision."""
    return RoutingDecision(
        outcome=RoutingOutcome.UNSUPPORTED_MODALITY,
        request_id=request_id,
        detected_modality=modality,
        confidence=confidence,
        adapter_key=None,
        reason=(
            f"Modality '{modality}' is not in the supported modality set. "
            f"Supported: {sorted(supported)}"
        ),
        recommended_action="verify_image_modality_and_retry",
        routing_time_ms=routing_time_ms,
        extra={"supported_modalities": sorted(supported)},
    )


def make_model_not_registered(
    request_id: str,
    modality: str,
    confidence: float,
    routing_time_ms: float = 0.0,
) -> RoutingDecision:
    """Factory: produce a MODEL_NOT_REGISTERED rejection decision."""
    return RoutingDecision(
        outcome=RoutingOutcome.MODEL_NOT_REGISTERED,
        request_id=request_id,
        detected_modality=modality,
        confidence=confidence,
        adapter_key=None,
        reason=(
            f"No adapter is registered in ModelRegistry for modality '{modality}'. "
            f"Register the adapter before routing requests."
        ),
        recommended_action="register_adapter_and_retry",
        routing_time_ms=routing_time_ms,
    )


def make_routing_failed(
    request_id: str,
    modality: str,
    confidence: float,
    error_message: str,
    routing_time_ms: float = 0.0,
) -> RoutingDecision:
    """Factory: produce a ROUTING_FAILED decision (unexpected error)."""
    return RoutingDecision(
        outcome=RoutingOutcome.ROUTING_FAILED,
        request_id=request_id,
        detected_modality=modality,
        confidence=confidence,
        adapter_key=None,
        reason=f"Routing failed with unexpected error: {error_message}",
        recommended_action="inspect_logs_and_retry",
        routing_time_ms=routing_time_ms,
        extra={"error_message": error_message},
    )
