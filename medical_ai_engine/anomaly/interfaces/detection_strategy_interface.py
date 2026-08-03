"""
detection_strategy_interface.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Abstract interface contract for all Anomaly Detection Strategies.

This module defines the plugin contract that every detection strategy must implement.
It does NOT contain any algorithmic logic; implementations live in Module 16.2+.

Plugin Architecture
-------------------
All detection strategies are registered in the AnomalyRegistry via their `strategy_key`.
The AnomalyDetectionEngine selects and dispatches to the appropriate strategy at runtime.

Extension Points
----------------
New strategies implement IDetectionStrategy and register with:
    registry.register(MyStrategy())

Supported Planned Strategies (Module 16.2+)
-------------------------------------------
  confidence_threshold   — Simple confidence comparison against thresholds.
  feature_distance       — Cosine / Euclidean distance from class prototype vectors.
  mahalanobis            — Mahalanobis distance from in-distribution Gaussian statistics.
  autoencoder            — Reconstruction error via trained autoencoder per modality.
  deep_svdd              — Deep SVDD one-class classification.
  openmax                — OpenMax probability correction for open-set recognition.
  energy_based           — Energy-based score from model logits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..anomaly_context import AnomalyContext


class IDetectionStrategy(ABC):
    """
    Abstract base class defining the plugin contract for all anomaly detection strategies.

    Lifecycle
    ---------
    1. Instantiated and registered at startup via AnomalyRegistry.register().
    2. Initialized (once) via initialize() with runtime dependencies.
    3. Called per-request via detect() for each incoming prediction.
    4. Cleaned up via cleanup() when the engine shuts down.

    Contract
    --------
    - detect() MUST be thread-safe (can be called concurrently).
    - detect() MUST NOT modify PredictionResult or EngineContext.
    - detect() populates relevant fields on AnomalyContext.
    - detect() MUST NOT raise; exceptions are caught by the engine pipeline.
    - detect() returns True if an anomaly was positively identified.

    Architecture Note
    -----------------
    This is a PURE ARCHITECTURE DEFINITION.
    All algorithmic implementations live in Module 16.2+ plugins.
    """

    @property
    @abstractmethod
    def strategy_key(self) -> str:
        """
        Unique registry key for this strategy.

        Convention: lowercase_snake_case.
        Examples: "confidence_threshold", "mahalanobis", "autoencoder"
        """

    @property
    @abstractmethod
    def strategy_version(self) -> str:
        """
        Semantic version string for this strategy implementation.
        Example: "1.0.0"
        """

    @property
    @abstractmethod
    def strategy_description(self) -> str:
        """
        Short human-readable description of how this strategy detects anomalies.
        """

    @property
    def requires_feature_vector(self) -> bool:
        """
        True if this strategy requires a feature vector from the model.
        Used by the AnomalyDetectionEngine to decide whether to extract features.
        Default: False (confidence-only strategies).
        """
        return False

    @property
    def requires_model(self) -> bool:
        """
        True if this strategy needs direct access to the trained model weights.
        Used to gate model-requiring strategies when model is unavailable.
        Default: False.
        """
        return False

    @abstractmethod
    def initialize(self, config: Dict[str, Any], model_adapter: Optional[Any] = None) -> None:
        """
        Initialize the strategy with runtime configuration and optional model adapter.

        Called once at startup by AnomalyRegistry after registration.

        Parameters
        ----------
        config : Dict[str, Any]
            Strategy-specific configuration parameters.
        model_adapter : Optional[Any]
            Optional trained model adapter reference for feature extraction.
        """

    @abstractmethod
    def detect(self, context: AnomalyContext) -> bool:
        """
        Execute anomaly detection for a single prediction request.

        Parameters
        ----------
        context : AnomalyContext
            Shared mutable execution context containing prediction inputs and partial results.
            This method populates its relevant fields (e.g., ood_analysis, reconstruction_error).

        Returns
        -------
        bool
            True if an anomaly was positively identified by this strategy.
            False if input is considered in-distribution / known.

        Thread Safety
        -------------
        MUST be thread-safe. Multiple concurrent requests may call detect() simultaneously.
        """

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return strategy metadata for registry inspection and health checks.

        Returns
        -------
        Dict containing at minimum:
            key: str
            version: str
            description: str
            requires_feature_vector: bool
            requires_model: bool
            is_initialized: bool
        """

    def cleanup(self) -> None:
        """
        Release resources held by this strategy (e.g., model weights, cached statistics).
        Called during engine shutdown. Default: no-op.
        """

    def health_check(self) -> Dict[str, Any]:
        """
        Return current health status of this strategy.
        Default implementation returns minimal status.
        """
        return {
            "strategy_key": self.strategy_key,
            "version": self.strategy_version,
            "status": "OK",
        }


__all__ = ["IDetectionStrategy"]
