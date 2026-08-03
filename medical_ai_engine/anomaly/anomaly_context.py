"""
anomaly_context.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Mutable shared execution context passed through the Anomaly Detection Pipeline.

Design Principles
-----------------
- Single shared state carrier, analogous to EngineContext in Module 14.
- Populated progressively by each detection sub-component.
- Never raises; errors and warnings are accumulated for audit.
- Does NOT import or depend on MedicalAIEngine or EngineContext (decoupled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class AnomalyContext:
    """
    Shared mutable execution context for the Anomaly Detection Pipeline.

    Fields are organized by lifecycle phase:
        Identity          → request_id, session_id
        Prediction Input  → modality, predicted_class, confidence, quality_score
        Feature Space     → feature_vector, reconstruction_error, feature_distance
        Sub-component     → confidence_analysis, ood_analysis, unknown_disease_analysis
        Risk              → risk_level, risk_score, risk_factors
        Decision          → anomaly_status, known_unknown_status, recommended_action
        Audit             → timestamps, logs, warnings, errors, performance_metrics
        Extensions        → future_extensions
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str
    session_id: str
    timestamp: str = field(default_factory=_utc_now)

    # ── Prediction Input (from PredictionResult / EngineContext) ──────────────
    modality: str = "unknown"
    predicted_class: str = "unknown"
    confidence: float = 0.0
    quality_score: Optional[float] = None
    """
    Image quality score (0.0–1.0).
    If None, quality check was not performed or unavailable.
    """

    # ── Feature Space (populated by strategy sub-components) ──────────────────
    feature_vector: Optional[Any] = None
    """
    Optional extracted feature vector (numpy array or tensor) from the model.
    Populated by feature-extraction-based strategies (Mahalanobis, Deep SVDD, etc.).
    None when using confidence-only strategies.
    """

    reconstruction_error: Optional[float] = None
    """
    Autoencoder reconstruction error for the input image.
    Populated only by autoencoder-based detection strategies (Module 16.x).
    """

    feature_distance: Optional[float] = None
    """
    Distance from the nearest known-class cluster in feature space.
    Populated by Mahalanobis or cosine-distance strategies.
    """

    openmax_scores: Optional[Dict[str, float]] = None
    """
    Per-class probability corrections from OpenMax strategy.
    Schema: {class_name: corrected_probability}
    """

    energy_score: Optional[float] = None
    """
    Energy-based anomaly score. Lower energy = more in-distribution.
    Populated by energy-based detection strategies.
    """

    # ── Sub-component Analysis Results ────────────────────────────────────────
    confidence_analysis: Optional[Dict[str, Any]] = None
    """
    Output of ConfidenceAnalyzer sub-component.
    Schema:
        {
            "confidence_score": float,
            "confidence_level": str,   # "HIGH" | "MODERATE" | "LOW" | "CRITICAL"
            "confidence_flag": str,    # "TRUSTED" | "LOW_CONFIDENCE" | "CRITICAL"
            "threshold_low": float,
            "threshold_critical": float,
        }
    """

    ood_analysis: Optional[Dict[str, Any]] = None
    """
    Output of OODDetector sub-component.
    Schema:
        {
            "is_ood": bool,
            "ood_score": float,         # higher = more OOD
            "distance": float,          # feature space distance
            "threshold": float,
            "strategy_used": str,
        }
    """

    unknown_disease_analysis: Optional[Dict[str, Any]] = None
    """
    Output of UnknownDiseaseDetector sub-component.
    Schema:
        {
            "is_unknown": bool,
            "reconstruction_error": float,
            "threshold": float,
            "strategy_used": str,
            "confidence_in_known": float,  # probability mass on known classes
        }
    """

    # ── Risk Assessment (populated by RiskAssessor) ───────────────────────────
    risk_level: str = "UNKNOWN"
    """
    Clinical risk level assigned after multi-signal fusion.
    Values: "LOW" | "MODERATE" | "HIGH" | "CRITICAL" | "UNKNOWN"
    """

    risk_score: float = 0.0
    """
    Composite numerical risk score (0.0–1.0).
    Derived from confidence, OOD score, reconstruction error, and quality score.
    """

    risk_factors: List[str] = field(default_factory=list)
    """
    Human-readable list of factors contributing to elevated risk.
    Examples: ["Low confidence (0.42)", "OOD score above threshold (3.7)"]
    """

    # ── Decision Engine Output ────────────────────────────────────────────────
    anomaly_status: str = "NOT_ANALYZED"
    """
    Primary anomaly detection outcome.
    Values:
        "KNOWN_DISEASE"    — Prediction is trusted; disease is recognized.
        "UNKNOWN_DISEASE"  — Features do not match any known disease pattern.
        "OUT_OF_DISTRIBUTION" — Input image is outside the model's training distribution.
        "LOW_CONFIDENCE"   — Model confidence below acceptable threshold.
        "POOR_QUALITY"     — Input image quality is insufficient for reliable prediction.
        "NOT_ANALYZED"     — Anomaly detection was skipped or disabled.
    """

    known_unknown_status: str = "UNKNOWN"
    """
    Binary known/unknown classification.
    Values: "KNOWN" | "UNKNOWN" | "UNCERTAIN"
    """

    recommended_action: str = "PROCEED"
    """
    Clinical action recommendation from the Decision Engine.
    Values:
        "PROCEED"              — Prediction is trustworthy; use result.
        "MANUAL_REVIEW"        — Flag for clinician manual review.
        "REPEAT_ACQUISITION"   — Request a new image acquisition.
        "ESCALATE"             — Escalate to specialist immediately.
        "DISCARD"              — Image or prediction is unusable.
    """

    decision_reason: str = ""
    """
    Human-readable explanation of the anomaly decision and recommended action.
    """

    detection_method: str = "not_applied"
    """
    Key of the detection strategy that produced the final decision.
    """

    # ── Execution State ───────────────────────────────────────────────────────
    execution_state: str = "CREATED"
    """
    Lifecycle state: CREATED → RUNNING → COMPLETED | FAILED
    """

    # ── Audit Trail ───────────────────────────────────────────────────────────
    timestamps: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # ── Extension Namespace ───────────────────────────────────────────────────
    future_extensions: Dict[str, Any] = field(default_factory=dict)
    """
    Open namespace for Module 16.2+ strategies to store custom data
    without changing AnomalyContext schema.
    Convention: use strategy-prefixed keys, e.g., "mahalanobis_covariance".
    """

    # ── Methods ───────────────────────────────────────────────────────────────

    def log(self, message: str, stage: str = "") -> None:
        prefix = f"[{stage}] " if stage else ""
        self.logs.append(f"{_utc_now()} {prefix}{message}")

    def warn(self, message: str, stage: str = "") -> None:
        prefix = f"[{stage}] " if stage else ""
        self.warnings.append(f"{_utc_now()} {prefix}WARNING: {message}")

    def add_error(self, stage: str, error: str) -> None:
        self.errors.append({"stage": stage, "error": error, "timestamp": _utc_now()})

    def set_timestamp(self, key: str) -> None:
        self.timestamps[key] = _utc_now()

    def record_metric(self, key: str, value_ms: float) -> None:
        self.performance_metrics[key] = round(value_ms, 3)

    def summary(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "modality": self.modality,
            "confidence": self.confidence,
            "anomaly_status": self.anomaly_status,
            "known_unknown_status": self.known_unknown_status,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 4),
            "recommended_action": self.recommended_action,
            "detection_method": self.detection_method,
            "execution_state": self.execution_state,
            "warnings": len(self.warnings),
            "errors": len(self.errors),
        }

    def __repr__(self) -> str:
        return (
            f"AnomalyContext("
            f"request_id={self.request_id!r}, "
            f"anomaly_status={self.anomaly_status!r}, "
            f"risk_level={self.risk_level!r})"
        )


__all__ = ["AnomalyContext"]
