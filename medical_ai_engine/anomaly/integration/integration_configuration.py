"""
integration_configuration.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Configuration dataclass for Anomaly Engine Integration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .integration_exceptions import IntegrationConfigurationError


@dataclass
class IntegrationConfiguration:
    """
    Configuration for Anomaly Engine Integration.

    Attributes
    ----------
    enabled : bool
        Master flag to enable or disable post-prediction anomaly detection.
    async_execution : bool
        If True, run anomaly detection in a background thread without blocking.
    fail_silently : bool
        If True, preserve prediction and explainability if anomaly detection fails.
    timeout_ms : float
        Maximum allowed latency in ms for anomaly detection execution.
    supported_modalities : List[str]
        Supported medical imaging modalities for anomaly detection.
    policy_name : str
        Active decision policy name.
    """
    enabled: bool = True
    async_execution: bool = False
    fail_silently: bool = True
    timeout_ms: float = 1000.0
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri",
            "chest_xray",
            "histopathology",
            "retina",
            "skin",
            "ultrasound",
            "brain_ct",
        ]
    )
    policy_name: str = "default_priority"

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise IntegrationConfigurationError(f"timeout_ms must be > 0; got {self.timeout_ms}")

    @classmethod
    def default(cls) -> "IntegrationConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "IntegrationConfiguration":
        return cls(timeout_ms=500.0)


__all__ = ["IntegrationConfiguration"]
