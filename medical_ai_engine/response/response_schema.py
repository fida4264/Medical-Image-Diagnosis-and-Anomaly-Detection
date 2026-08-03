"""
Medical AI Engine — Response Schema

Defines enumerations, status codes, and schema constants for PredictionResult.
"""

from __future__ import annotations

from enum import Enum


class PredictionStatus(str, Enum):
    """Final status of a completed prediction request."""

    SUCCESS = "SUCCESS"
    """Pipeline executed fully; prediction result is complete."""

    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    """Pipeline executed fully but non-fatal warnings were recorded."""

    FAILED = "FAILED"
    """Pipeline execution failed; prediction result contains error info."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    """Pipeline completed but modality confidence was below threshold."""

    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    """Pipeline rejected due to unrecognised image modality."""

    MODEL_NOT_REGISTERED = "MODEL_NOT_REGISTERED"
    """No adapter registered in ModelRegistry for the detected modality."""


class PredictionReliability(str, Enum):
    """Reliability grade of the disease prediction confidence score."""

    HIGH = "high"
    """Confidence ≥ high_threshold (default 0.85)."""

    MODERATE = "moderate"
    """Confidence ≥ moderate_threshold (default 0.70) and < high_threshold."""

    LOW = "low"
    """Confidence < moderate_threshold."""

    UNKNOWN = "unknown"
    """Reliability could not be determined (e.g. pipeline failed)."""


# Supported serialization formats
SERIALIZATION_FORMATS = ("json", "pretty_json", "dict")

# Reserved for future FHIR interoperability — not implemented
RESERVED_FORMATS = ("fhir_r4",)

# Schema version for forward-compatibility tracking
RESPONSE_SCHEMA_VERSION = "1.0.0"

# Required top-level PredictionResult fields
REQUIRED_FIELDS = (
    "request_id",
    "session_id",
    "timestamp",
    "modality",
    "predicted_disease",
    "confidence",
    "status",
)
