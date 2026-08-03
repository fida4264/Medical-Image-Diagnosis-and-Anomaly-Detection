"""
integration_configuration.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Configuration dataclass for the Explainability Integration subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .integration_exceptions import IntegrationConfigurationError


@dataclass
class IntegrationConfiguration:
    """
    Configuration for the Module 15.6 Explainability Engine Integration layer.

    Attributes
    ----------
    enabled:
        Master switch. If False the entire integration layer is bypassed.
    async_execution:
        Run explainability in a daemon background thread (non-blocking).
    min_confidence_threshold:
        Predictions with confidence below this value skip explainability.
    cam_method:
        Default CAM method key to use (must be registered in CAMProductionRegistry).
    export_artifacts:
        Persist heatmap / overlay images to disk.
    artifact_output_dir:
        Directory to save generated artifacts.
    include_clinical_explanation:
        Run the Module 15.5 ClinicalExplanationEngine after visualization.
    max_concurrent_explanations:
        Thread pool size cap for concurrent explainability executions.
    timeout_seconds:
        Synchronous execution timeout. 0 = no timeout.
    fail_silently:
        If True, prediction is never failed by explainability errors.
    supported_modalities:
        Modalities to process; empty list means all modalities.
    """

    enabled: bool = True
    async_execution: bool = False          # sync by default for test stability
    min_confidence_threshold: float = 0.30
    cam_method: str = "gradcam"
    export_artifacts: bool = False
    artifact_output_dir: str = "artifacts/xai"
    include_clinical_explanation: bool = True
    max_concurrent_explanations: int = 4
    timeout_seconds: float = 30.0
    fail_silently: bool = True
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri", "chest_xray", "brain_ct",
            "histopathology", "retina", "skin", "ultrasound",
        ]
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise IntegrationConfigurationError(
                f"min_confidence_threshold must be in [0, 1]; got {self.min_confidence_threshold}"
            )
        if self.max_concurrent_explanations < 1:
            raise IntegrationConfigurationError(
                f"max_concurrent_explanations must be ≥ 1; got {self.max_concurrent_explanations}"
            )
        if self.timeout_seconds < 0:
            raise IntegrationConfigurationError(
                f"timeout_seconds must be ≥ 0; got {self.timeout_seconds}"
            )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "async_execution": self.async_execution,
            "min_confidence_threshold": self.min_confidence_threshold,
            "cam_method": self.cam_method,
            "export_artifacts": self.export_artifacts,
            "artifact_output_dir": self.artifact_output_dir,
            "include_clinical_explanation": self.include_clinical_explanation,
            "max_concurrent_explanations": self.max_concurrent_explanations,
            "timeout_seconds": self.timeout_seconds,
            "fail_silently": self.fail_silently,
        }

    @classmethod
    def default(cls) -> "IntegrationConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "IntegrationConfiguration":
        """Lightweight configuration for unit tests."""
        return cls(
            enabled=True,
            async_execution=False,
            min_confidence_threshold=0.0,
            export_artifacts=False,
            include_clinical_explanation=True,
            fail_silently=True,
        )


__all__ = ["IntegrationConfiguration"]
