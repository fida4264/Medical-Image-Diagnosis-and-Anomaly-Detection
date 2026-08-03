"""
Medical AI Engine — Response Configuration

Configures ResponseBuilder behavior: serialization format, validation rules,
clinical report structure, and metadata population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResponseConfiguration:
    """
    Configuration for ResponseBuilder and clinical report aggregation.
    """

    # Serialization
    default_serialization_format: str = "json"
    """Default serialization output format: 'json', 'pretty_json', or 'dict'."""

    # Validation settings
    strict_validation: bool = True
    """If True, raise ResponseValidationException on any validation error."""

    require_disease_prediction: bool = True
    """If True, validate that disease_prediction is present in ctx."""

    require_clinical_report: bool = True
    """If True, validate that clinical_report is present in ctx."""

    # Clinical report settings
    include_differential_diagnoses: bool = True
    """Include top-K differential diagnoses in clinical report aggregation."""

    max_differential_count: int = 5
    """Maximum differential diagnosis entries to include in clinical report."""

    include_clinical_note: bool = True
    """Include auto-generated clinical note in aggregated report."""

    include_severity_grade: bool = True
    """Include severity grade when available (e.g. retina Module 11)."""

    # Metadata settings
    include_performance_metrics: bool = True
    """Include pipeline performance metrics in PredictionResult."""

    include_model_metadata: bool = True
    """Include adapter/model metadata in PredictionResult."""

    include_routing_decision: bool = True
    """Include routing decision details in PredictionResult."""

    # Extension hooks
    include_gradcam_if_present: bool = True
    """Include Grad-CAM result if populated by Module 15."""

    include_anomaly_if_present: bool = True
    """Include anomaly detection result if populated by Module 16."""

    # Reliability thresholds
    high_reliability_threshold: float = 0.85
    """Confidence score at or above which reliability = 'high'."""

    moderate_reliability_threshold: float = 0.70
    """Confidence score at or above which reliability = 'moderate'."""

    # Future FHIR interoperability (reserved, not yet implemented)
    enable_fhir_output: bool = False
    """Reserved: FHIR R4 output support (not implemented in this version)."""

    @classmethod
    def default(cls) -> "ResponseConfiguration":
        """Return standard default configuration."""
        return cls()

    @classmethod
    def strict(cls) -> "ResponseConfiguration":
        """Return strict validation configuration."""
        return cls(strict_validation=True, require_disease_prediction=True, require_clinical_report=True)
