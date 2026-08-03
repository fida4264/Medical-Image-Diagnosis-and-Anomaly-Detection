"""
anomaly_result.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Immutable standardized output of the Anomaly Detection Engine.

Design Principles
-----------------
- Frozen dataclass: no field can be modified after construction.
- JSON-serializable via to_dict() / to_json().
- Contains only the final decision — no raw intermediate signals.
- Conforms to the AnomalyResult JSON Schema defined in Module 16.1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AnomalyStatus(str, Enum):
    """
    Standardized anomaly detection outcome values.

    Values
    ------
    KNOWN_DISEASE
        Prediction is trusted; the image belongs to a recognized disease category.
    UNKNOWN_DISEASE
        Features deviate from all known disease patterns; novel case detected.
    OUT_OF_DISTRIBUTION
        Input image is outside the model's training distribution entirely.
    LOW_CONFIDENCE
        Model confidence is below the acceptable clinical threshold.
    POOR_QUALITY
        Input image quality is insufficient for reliable detection.
    NOT_ANALYZED
        Anomaly detection was skipped (disabled, modality not supported, etc.).
    """

    KNOWN_DISEASE = "KNOWN_DISEASE"
    UNKNOWN_DISEASE = "UNKNOWN_DISEASE"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    POOR_QUALITY = "POOR_QUALITY"
    NOT_ANALYZED = "NOT_ANALYZED"


class RiskLevel(str, Enum):
    """Clinical risk classification levels."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RecommendedAction(str, Enum):
    """
    Clinical action recommendations produced by the Decision Engine.

    PROCEED
        Prediction is trustworthy; proceed with clinical workflow.
    MANUAL_REVIEW
        Flag for clinician manual review before clinical use.
    REPEAT_ACQUISITION
        Request a new image acquisition due to quality or uncertainty issues.
    REPEAT_SCAN
        Alias for REPEAT_ACQUISITION.
    ESCALATE
        Escalate to specialist immediately; high-risk or unknown finding.
    DISCARD
        Image or prediction is unusable; do not use in clinical workflow.
    REJECT_PREDICTION
        Alias for DISCARD / rejection of clinical prediction.
    """

    PROCEED = "PROCEED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REPEAT_ACQUISITION = "REPEAT_ACQUISITION"
    REPEAT_SCAN = "REPEAT_SCAN"
    ESCALATE = "ESCALATE"
    DISCARD = "DISCARD"
    REJECT_PREDICTION = "REJECT_PREDICTION"


@dataclass(frozen=True)
class AnomalyResult:
    """
    Immutable, standardized output of the Anomaly Detection Engine.

    Produced after a full anomaly detection pipeline execution.
    Attached to PredictionResult.extra["anomaly_result"] (Module 16.x).

    Fields
    ------
    request_id : str
        Links this result to the originating prediction request.
    session_id : str
        Links this result to the originating execution session.
    timestamp : str
        UTC ISO 8601 timestamp of result construction.

    status : AnomalyStatus
        Primary anomaly detection outcome.
    risk_level : RiskLevel
        Clinical risk classification assigned by RiskAssessor.
    known_unknown_status : str
        Binary known/unknown classification: "KNOWN" | "UNKNOWN" | "UNCERTAIN".
    confidence_score : float
        Original prediction confidence score (0.0–1.0).
    risk_score : float
        Composite numerical risk score (0.0–1.0).
    recommended_action : RecommendedAction
        Actionable clinical recommendation.
    reason : str
        Human-readable explanation of the detection decision.
    detection_method : str
        Strategy key that produced the final decision.
    processing_time_ms : float
        Total anomaly detection pipeline execution time in milliseconds.

    risk_factors : List[str]
        Human-readable list of contributing risk factors.
    confidence_analysis : Optional[Dict]
        Raw confidence analysis sub-component output.
    ood_analysis : Optional[Dict]
        Raw OOD detection sub-component output.
    unknown_disease_analysis : Optional[Dict]
        Raw unknown disease detection sub-component output.

    warnings : List[str]
        Non-fatal warnings accumulated during detection.
    errors : List[Dict[str, str]]
        Non-fatal errors recorded during detection.
    metadata : Dict[str, Any]
        Open extension namespace for strategy-specific metadata.
    """

    # ── Identity & Timestamp ──────────────────────────────────────────────────
    request_id: str
    session_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ── Core Decision Fields ──────────────────────────────────────────────────
    status: AnomalyStatus = AnomalyStatus.NOT_ANALYZED
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    known_unknown_status: str = "UNCERTAIN"
    confidence_score: float = 0.0
    risk_score: float = 0.0
    recommended_action: RecommendedAction = RecommendedAction.MANUAL_REVIEW
    reason: str = ""
    detection_method: str = "not_applied"
    processing_time_ms: float = 0.0

    # ── Supporting Evidence ───────────────────────────────────────────────────
    risk_factors: List[str] = field(default_factory=list)
    confidence_analysis: Optional[Dict[str, Any]] = None
    ood_analysis: Optional[Dict[str, Any]] = None
    unknown_disease_analysis: Optional[Dict[str, Any]] = None

    # ── Audit Trail ───────────────────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def is_trusted(self) -> bool:
        """True if the prediction is deemed trustworthy for clinical use."""
        return self.status == AnomalyStatus.KNOWN_DISEASE and self.risk_level in (
            RiskLevel.LOW, RiskLevel.MODERATE
        )

    @property
    def requires_review(self) -> bool:
        """True if human review is required before clinical use."""
        return self.recommended_action in (
            RecommendedAction.MANUAL_REVIEW,
            RecommendedAction.ESCALATE,
        )

    @property
    def is_anomalous(self) -> bool:
        """True if an anomaly condition was detected."""
        return self.status not in (
            AnomalyStatus.KNOWN_DISEASE,
            AnomalyStatus.NOT_ANALYZED,
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        d["risk_level"] = self.risk_level.value
        d["recommended_action"] = self.recommended_action.value
        return d

    def to_json(self) -> str:
        """Serialize to compact JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_pretty_json(self) -> str:
        """Serialize to human-readable indented JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"AnomalyResult("
            f"request_id={self.request_id!r}, "
            f"status={self.status.value!r}, "
            f"risk_level={self.risk_level.value!r}, "
            f"recommended_action={self.recommended_action.value!r})"
        )


__all__ = ["AnomalyResult", "AnomalyStatus", "RiskLevel", "RecommendedAction"]
