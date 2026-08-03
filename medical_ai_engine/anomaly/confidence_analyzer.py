"""
confidence_analyzer.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Confidence-based trust scorer for prediction reliability.

Architecture Role
-----------------
The ConfidenceAnalyzer is the first-pass filter in the anomaly detection pipeline.
It evaluates the model's confidence score against configurable thresholds
to determine if the prediction is minimally trustworthy before deeper analysis.

This is a PURE ARCHITECTURE DEFINITION.
No ML algorithms are implemented here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_exceptions import ConfidenceAnalysisError

logger = logging.getLogger(__name__)


class ConfidenceAnalyzer:
    """
    Analyzes prediction confidence and classifies the confidence level.

    Confidence Classification Schema
    ---------------------------------
    confidence >= threshold_low         → TRUSTED / HIGH_CONFIDENCE
    threshold_critical <= c < threshold_low  → LOW_CONFIDENCE
    confidence < threshold_critical     → CRITICAL

    Pipeline Integration
    --------------------
    Called as Step 1 in AnomalyDetectionEngine pipeline.
    Populates `context.confidence_analysis`.
    """

    CONFIDENCE_LEVELS = {
        "HIGH": "HIGH_CONFIDENCE",
        "LOW": "LOW_CONFIDENCE",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, config: AnomalyConfiguration) -> None:
        self._config = config
        logger.debug(
            "[ConfidenceAnalyzer] Initialized (low_threshold=%.2f, critical_threshold=%.2f).",
            config.confidence_threshold_low,
            config.confidence_threshold_critical,
        )

    def analyze(self, context: AnomalyContext) -> None:
        """
        Analyze confidence and populate context.confidence_analysis.

        Parameters
        ----------
        context : AnomalyContext
            Shared execution context. confidence field must be set.
        """
        t0 = time.perf_counter()

        try:
            confidence = context.confidence
            low_threshold = self._config.confidence_threshold_low
            critical_threshold = self._config.confidence_threshold_critical

            # Classify confidence level
            if confidence >= low_threshold:
                confidence_level = "HIGH"
                confidence_flag = "TRUSTED"
            elif confidence >= critical_threshold:
                confidence_level = "LOW"
                confidence_flag = "LOW_CONFIDENCE"
            else:
                confidence_level = "CRITICAL"
                confidence_flag = "CRITICAL"

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            context.confidence_analysis = {
                "confidence_score": round(confidence, 4),
                "confidence_level": confidence_level,
                "confidence_flag": confidence_flag,
                "threshold_low": low_threshold,
                "threshold_critical": critical_threshold,
                "analysis_time_ms": round(elapsed_ms, 3),
            }

            context.record_metric("confidence_analysis_ms", elapsed_ms)
            context.log(
                f"Confidence analysis: score={confidence:.4f}, "
                f"level={confidence_level}, flag={confidence_flag}",
                stage="ConfidenceAnalyzer",
            )

            if confidence_flag != "TRUSTED":
                context.warn(
                    f"Prediction confidence ({confidence:.4f}) is {confidence_flag}.",
                    stage="ConfidenceAnalyzer",
                )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.add_error("ConfidenceAnalyzer", str(exc))
            context.warn("Confidence analysis failed; defaulting to UNKNOWN trust.", stage="ConfidenceAnalyzer")
            logger.exception("[ConfidenceAnalyzer] Analysis failed: %s", exc)
            # Architecture: never raise; caller handles None context.confidence_analysis
            context.confidence_analysis = {
                "confidence_score": context.confidence,
                "confidence_level": "UNKNOWN",
                "confidence_flag": "UNKNOWN",
                "error": str(exc),
            }

    def get_config_summary(self) -> Dict[str, Any]:
        return {
            "low_threshold": self._config.confidence_threshold_low,
            "critical_threshold": self._config.confidence_threshold_critical,
        }


__all__ = ["ConfidenceAnalyzer"]
