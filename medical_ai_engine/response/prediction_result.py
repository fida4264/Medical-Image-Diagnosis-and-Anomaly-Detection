"""
Medical AI Engine — PredictionResult

Immutable (frozen=True) dataclass representing the final standardized
prediction response returned by the MedicalAIEngine after pipeline execution.

After construction no field may be modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from medical_ai_engine.response.response_schema import (
    PredictionStatus,
    PredictionReliability,
    RESPONSE_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable, standardized prediction response for any completed Medical AI request.

    This is the final output of the MedicalAIEngine pipeline — produced by
    ResponseBuilder and returned to the caller. No field may be modified
    after construction (frozen=True enforces this).

    Fields:
        request_id           — Unique request identifier (UUID4)
        session_id           — Execution session identifier
        timestamp            — UTC ISO 8601 timestamp of result construction
        schema_version       — Response schema version for forward-compatibility
        status               — PredictionStatus enum value
        modality             — Detected image modality (e.g. "chest_xray")
        predicted_disease    — Primary diagnosed condition (str)
        confidence           — Disease prediction confidence score (0.0–1.0)
        prediction_reliability  — PredictionReliability grade
        requires_manual_review  — Flag indicating clinician review needed
        routing_decision     — Routing decision metadata dict
        clinical_report      — Aggregated clinical report dict
        performance_metrics  — Per-stage timing metrics dict
        model_metadata       — Model/adapter metadata dict
        warnings             — Non-fatal pipeline warnings list
        errors               — Non-fatal pipeline error records list
        top_predictions      — Top-K differential disease predictions list
        severity_grade       — Severity integer when available (e.g. retina)
        gradcam_result       — Grad-CAM heatmap metadata (Module 15 hook)
        anomaly_result       — Anomaly detection result (Module 16 hook)
        extra                — Open extension namespace for future modules
    """

    # ── Identity & Timestamps ─────────────────────────────────────────────────
    request_id: str
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    schema_version: str = RESPONSE_SCHEMA_VERSION

    # ── Status & Routing ──────────────────────────────────────────────────────
    status: PredictionStatus = PredictionStatus.SUCCESS
    modality: str = "unknown"
    routing_decision: Dict[str, Any] = field(default_factory=dict)

    # ── Disease Prediction ────────────────────────────────────────────────────
    predicted_disease: str = "unknown"
    confidence: float = 0.0
    prediction_reliability: PredictionReliability = PredictionReliability.UNKNOWN
    requires_manual_review: bool = False
    top_predictions: List[Dict[str, Any]] = field(default_factory=list)
    severity_grade: Optional[int] = None

    # ── Clinical Report ───────────────────────────────────────────────────────
    clinical_report: Dict[str, Any] = field(default_factory=dict)

    # ── Performance & Metadata ────────────────────────────────────────────────
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Audit Trail ───────────────────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    # ── Extension Hooks ───────────────────────────────────────────────────────
    gradcam_result: Optional[Dict[str, Any]] = None
    anomaly_result: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_success(self) -> bool:
        return self.status in (
            PredictionStatus.SUCCESS,
            PredictionStatus.COMPLETED_WITH_WARNINGS,
        )

    @property
    def is_failed(self) -> bool:
        return self.status == PredictionStatus.FAILED

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert PredictionResult to a JSON-serializable dictionary."""
        d = asdict(self)
        # Serialize enum fields to their string values
        d["status"] = self.status.value
        d["prediction_reliability"] = self.prediction_reliability.value
        return d

    def to_json(self) -> str:
        """Serialize PredictionResult to compact JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_pretty_json(self) -> str:
        """Serialize PredictionResult to indented, human-readable JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"PredictionResult("
            f"request_id={self.request_id!r}, "
            f"status={self.status.value!r}, "
            f"modality={self.modality!r}, "
            f"predicted_disease={self.predicted_disease!r}, "
            f"confidence={self.confidence:.3f})"
        )
