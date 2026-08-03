"""
ood_detector.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Out-of-Distribution (OOD) detection facade.

Architecture Role
-----------------
The OODDetector is the second-stage component in the anomaly detection pipeline.
It determines whether the input image belongs outside the model's training distribution.

ARCHITECTURE ONLY — No algorithm implemented.
Strategy implementations live in Module 16.2+ (Mahalanobis, Energy, Deep SVDD, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_exceptions import OODDetectionError
from .anomaly_registry import AnomalyRegistry

logger = logging.getLogger(__name__)

# Placeholder sentinel when no OOD strategy is available in the registry.
_NO_STRATEGY_SENTINEL = "__no_ood_strategy__"


class OODDetector:
    """
    Facade that dispatches OOD detection to the registered detection strategy.

    Pipeline Integration
    --------------------
    Called as Step 2 in AnomalyDetectionEngine pipeline.
    Populates context.ood_analysis.

    Extension Points
    ----------------
    Any strategy registered in AnomalyRegistry with a key listed in
    `ood_strategy_keys` can be used as the OOD detector.
    The default order preference is:
        mahalanobis > energy_based > feature_distance > confidence_threshold

    Failure Policy
    --------------
    If the strategy fails, OOD analysis is recorded as UNKNOWN (not OOD).
    Primary prediction is never disrupted.
    """

    # Ordered preference of OOD strategy keys (first available is used)
    OOD_STRATEGY_PREFERENCE = [
        "mahalanobis",
        "energy_based",
        "deep_svdd",
        "feature_distance",
        "confidence_threshold",
    ]

    def __init__(self, config: AnomalyConfiguration, registry: AnomalyRegistry) -> None:
        self._config = config
        self._registry = registry
        logger.debug("[OODDetector] Initialized (threshold=%.2f).", config.ood_distance_threshold)

    def detect(self, context: AnomalyContext) -> bool:
        """
        Run OOD detection and populate context.ood_analysis.

        Returns
        -------
        bool
            True if image is detected as OOD.
        """
        t0 = time.perf_counter()
        is_ood = False

        try:
            strategy_key, strategy = self._select_strategy()

            if strategy is None:
                # No OOD strategy available — default to NOT OOD
                context.ood_analysis = {
                    "is_ood": False,
                    "ood_score": 0.0,
                    "distance": None,
                    "threshold": self._config.ood_distance_threshold,
                    "strategy_used": "none",
                    "status": "NO_STRATEGY_AVAILABLE",
                }
                context.warn(
                    "No OOD strategy registered; skipping OOD detection.",
                    stage="OODDetector",
                )
                return False

            # Delegate to the plugin strategy
            is_ood = strategy.detect(context)

            # Ensure ood_analysis was populated by the strategy
            if context.ood_analysis is None:
                context.ood_analysis = {
                    "is_ood": is_ood,
                    "ood_score": 0.0,
                    "distance": context.feature_distance,
                    "threshold": self._config.ood_distance_threshold,
                    "strategy_used": strategy_key,
                }

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.add_error("OODDetector", str(exc))
            context.ood_analysis = {
                "is_ood": False,
                "ood_score": 0.0,
                "distance": None,
                "threshold": self._config.ood_distance_threshold,
                "strategy_used": "error",
                "error": str(exc),
            }
            logger.exception("[OODDetector] OOD detection failed: %s", exc)
            is_ood = False

        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.record_metric("ood_detection_ms", elapsed_ms)
            context.log(
                f"OOD detection complete: is_ood={is_ood} ({elapsed_ms:.2f} ms)",
                stage="OODDetector",
            )

        return is_ood

    def _select_strategy(self):
        """Select the first available OOD strategy from preference list."""
        for key in self.OOD_STRATEGY_PREFERENCE:
            if self._registry.is_registered(key):
                return key, self._registry.get_strategy(key)
        return None, None


__all__ = ["OODDetector"]
