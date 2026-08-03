"""
unknown_disease_detector.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Unknown Disease / Novel Pathology detection facade.

Architecture Role
-----------------
The UnknownDiseaseDetector is Stage 3 of the anomaly detection pipeline.
It determines whether the input exhibits features of a disease that was NOT
present in the model's training set — i.e., a genuinely novel pathology.

Distinction from OOD
---------------------
- OOD: Image is OUTSIDE the training distribution entirely (wrong modality, artifact, etc.)
- Unknown Disease: Image IS inside the distribution (same modality type) but
  the disease pattern does not match any known class prototype.

ARCHITECTURE ONLY — No algorithm implemented.
Strategy implementations live in Module 16.2+ (autoencoder reconstruction, OpenMax, etc.).
"""

from __future__ import annotations

import logging
import time

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_registry import AnomalyRegistry

logger = logging.getLogger(__name__)


class UnknownDiseaseDetector:
    """
    Facade for detecting novel / unknown disease patterns not seen during training.

    Pipeline Integration
    --------------------
    Called as Step 3 in AnomalyDetectionEngine pipeline.
    Populates context.unknown_disease_analysis.

    Extension Points
    ----------------
    Strategy preference for unknown disease detection:
        autoencoder > openmax > feature_distance > confidence_threshold
    """

    UNKNOWN_STRATEGY_PREFERENCE = [
        "autoencoder",
        "openmax",
        "feature_distance",
        "confidence_threshold",
    ]

    def __init__(self, config: AnomalyConfiguration, registry: AnomalyRegistry) -> None:
        self._config = config
        self._registry = registry
        logger.debug(
            "[UnknownDiseaseDetector] Initialized (reconstruction_threshold=%.4f).",
            config.reconstruction_error_threshold,
        )

    def detect(self, context: AnomalyContext) -> bool:
        """
        Run unknown disease detection and populate context.unknown_disease_analysis.

        Returns
        -------
        bool
            True if an unknown/novel disease pattern is detected.
        """
        t0 = time.perf_counter()
        is_unknown = False

        try:
            strategy_key, strategy = self._select_strategy()

            if strategy is None:
                context.unknown_disease_analysis = {
                    "is_unknown": False,
                    "reconstruction_error": context.reconstruction_error,
                    "threshold": self._config.reconstruction_error_threshold,
                    "strategy_used": "none",
                    "status": "NO_STRATEGY_AVAILABLE",
                    "confidence_in_known": context.confidence,
                }
                context.warn(
                    "No unknown disease strategy registered; skipping.",
                    stage="UnknownDiseaseDetector",
                )
                return False

            is_unknown = strategy.detect(context)

            if context.unknown_disease_analysis is None:
                context.unknown_disease_analysis = {
                    "is_unknown": is_unknown,
                    "reconstruction_error": context.reconstruction_error,
                    "threshold": self._config.reconstruction_error_threshold,
                    "strategy_used": strategy_key,
                    "confidence_in_known": context.confidence,
                }

        except Exception as exc:
            context.add_error("UnknownDiseaseDetector", str(exc))
            context.unknown_disease_analysis = {
                "is_unknown": False,
                "reconstruction_error": None,
                "threshold": self._config.reconstruction_error_threshold,
                "strategy_used": "error",
                "error": str(exc),
                "confidence_in_known": context.confidence,
            }
            logger.exception("[UnknownDiseaseDetector] Detection failed: %s", exc)
            is_unknown = False

        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.record_metric("unknown_disease_detection_ms", elapsed_ms)
            context.log(
                f"Unknown disease detection complete: is_unknown={is_unknown} ({elapsed_ms:.2f} ms)",
                stage="UnknownDiseaseDetector",
            )

        return is_unknown

    def _select_strategy(self):
        for key in self.UNKNOWN_STRATEGY_PREFERENCE:
            if self._registry.is_registered(key):
                return key, self._registry.get_strategy(key)
        return None, None


__all__ = ["UnknownDiseaseDetector"]
