"""
Medical AI Engine — Integration Configuration

Configures settings for end-to-end system integration, including adapter registration,
validation rules, default serialization formats, and metric tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


SUPPORTED_MODALITIES_LIST = [
    "brain_mri",
    "chest_xray",
    "brain_ct",
    "histopathology",
    "retina",
    "skin",
    "ultrasound",
]


@dataclass
class IntegrationConfiguration:
    """
    Configuration options for system integration and workflow execution.
    """

    # Adapter Registration
    auto_register_adapters: bool = True
    """Automatically register predictor adapters for all 7 supported modalities on boot."""

    use_mock_fallbacks_if_missing: bool = True
    """
    If True, allows registering lightweight mock adapters when model weight files
    are missing from disk (useful for standalone testing / offline runs).
    """

    supported_modalities: List[str] = field(
        default_factory=lambda: list(SUPPORTED_MODALITIES_LIST)
    )
    """List of supported disease modalities."""

    # Workflow Execution
    default_serialization_format: str = "json"
    """Default output serialization format ('json', 'pretty_json', 'dict')."""

    enable_strict_validation: bool = True
    """If True, runs IntegrationValidator on every completed workflow result."""

    record_performance_metrics: bool = True
    """If True, collects latency breakdown in IntegrationMetrics."""

    max_workflow_timeout_seconds: float = 30.0
    """Maximum allowed duration for a single end-to-end prediction workflow."""

    min_modality_confidence_threshold: float = 0.50
    """Modality confidence score below which low_confidence status is assigned."""

    # Failure Recovery
    catch_pipeline_exceptions: bool = True
    """If True, catches pipeline exceptions and returns a FAILED PredictionResult cleanly."""

    @classmethod
    def default(cls) -> IntegrationConfiguration:
        """Return standard default integration configuration."""
        return cls()

    @classmethod
    def test_config(cls) -> IntegrationConfiguration:
        """Return integration configuration optimized for testing."""
        return cls(
            auto_register_adapters=True,
            use_mock_fallbacks_if_missing=True,
            enable_strict_validation=True,
            record_performance_metrics=True,
        )
