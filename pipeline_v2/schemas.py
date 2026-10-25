"""
pipeline_v2/schemas.py
======================
Pydantic V2 and dataclass schemas for Phase 9 structured inference,
quality gates, and decision hardening.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QualityGateState(str, Enum):
    """
    Standardized categorical decision states for V2.0 Phase 9 inference quality gates.
    """
    ACCEPTED_FOR_MODEL_OUTPUT = "ACCEPTED_FOR_MODEL_OUTPUT"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"
    REJECTED_INVALID_INPUT = "REJECTED_INVALID_INPUT"
    REJECTED_UNSUPPORTED_MODALITY = "REJECTED_UNSUPPORTED_MODALITY"
    REJECTED_OOD = "REJECTED_OOD"
    REVIEW_REQUIRED_ANOMALY = "REVIEW_REQUIRED_ANOMALY"
    INFERENCE_ERROR = "INFERENCE_ERROR"


class GateOutcome(BaseModel):
    """Individual quality gate evaluation record."""
    gate_name: str = Field(description="Identifier of the specific quality check")
    passed: bool = Field(description="Whether the quality gate passed")
    message: str = Field(description="Clinical/technical rationale for pass/flag/rejection")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic parameters")


class QualityGateResult(BaseModel):
    """Aggregated quality gate decision and rationale."""
    overall_status: QualityGateState = Field(description="Final operational decision status")
    rejection_reasons: List[str] = Field(default_factory=list, description="Reasons for rejection if abstained")
    review_reasons: List[str] = Field(default_factory=list, description="Reasons for manual review flag if applicable")
    gate_outcomes: Dict[str, GateOutcome] = Field(default_factory=dict, description="Outcomes of all checked gates")


class StructuredInferenceResponse(BaseModel):
    """
    Comprehensive, versioned structured inference response schema for V2.0 Phase 9.
    Separates raw model evidence, calibrated outputs, OOD detection, anomaly detection,
    and quality gate abstention status.
    """
    request_id: str = Field(description="Unique session/request tracking UUID")
    pipeline_version: str = Field(default="V2.0-Phase9", description="Software pipeline version")
    inference_timestamp: str = Field(description="ISO 8601 UTC timestamp of execution")

    # Modality and model traceability
    modality: str = Field(description="Target imaging modality")
    model_identifier: str = Field(description="Model architecture name")
    checkpoint_identifier: str = Field(description="Model weights checkpoint path or hash")

    # Decision status
    prediction_status: QualityGateState = Field(description="Primary quality-gate decision state")
    is_abstained: bool = Field(description="True if quality gates abstained from standard diagnostic classification")

    # Classification predictions
    predicted_class: Optional[str] = Field(default=None, description="Primary predicted class (null if abstained)")
    confidence_score: Optional[float] = Field(default=None, description="Calibrated top confidence (null if abstained)")
    raw_model_logits: Optional[Dict[str, float]] = Field(default=None, description="Unscaled raw model logits")
    calibrated_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Temperature-calibrated class probabilities")

    # Multi-label Retina specific fields
    retina_label_predictions: Optional[Dict[str, int]] = Field(
        default=None, description="Binary classification {0, 1} per Retina label using Phase 7 thresholds"
    )
    retina_thresholds_used: Optional[Dict[str, float]] = Field(
        default=None, description="Per-label optimal decision thresholds applied to Retina sigmoid outputs"
    )

    # Phase 8 Out-of-Distribution (OOD) Metrics
    ood_method: str = Field(description="Selected OOD detection scoring method")
    ood_score: Optional[float] = Field(default=None, description="OOD score (higher = more unfamiliar)")
    ood_threshold: Optional[float] = Field(default=None, description="Frozen operating threshold from validation split")
    ood_status: str = Field(description="OOD assessment: 'WITHIN_DISTRIBUTION', 'POTENTIALLY_OOD', or 'UNAVAILABLE'")

    # Phase 8 Anomaly Scoring Metrics (Within-Distribution Normal Deviation)
    anomaly_method: Optional[str] = Field(default=None, description="Reference distribution anomaly scoring method")
    anomaly_score: Optional[float] = Field(default=None, description="Distance from normal reference state")
    anomaly_threshold: Optional[float] = Field(default=None, description="90% specificity threshold from validation split")
    anomaly_status: str = Field(default="UNAVAILABLE", description="'NORMAL_REFERENCE_ALIGNED', 'POTENTIAL_ABNORMALITY', or 'UNAVAILABLE'")

    # Quality Gate Summary
    quality_gate_results: QualityGateResult = Field(description="Outcomes of all checked quality gates")
    rejection_or_review_reasons: List[str] = Field(default_factory=list, description="Consolidated review or rejection reasons")

    # Provenance and configuration
    preprocessing_configuration: Dict[str, Any] = Field(default_factory=dict, description="Image transforms and input specs")
    calibration_configuration: Dict[str, Any] = Field(default_factory=dict, description="Temperature or threshold parameters applied")
    limitations_notice: str = Field(description="Mandatory investigational research disclaimer")
