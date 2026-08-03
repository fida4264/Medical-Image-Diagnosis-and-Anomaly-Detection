"""
confidence_threshold.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Production ConfidenceThresholdStrategy implementing IDetectionStrategy.

Strategy Key
------------
"confidence_threshold"

Functionality
-------------
Evaluates prediction confidence against per-modality thresholds:
  - HIGH   (>= 0.85) → TRUSTED / KNOWN_DISEASE
  - MEDIUM (>= 0.70) → TRUSTED / MODERATE
  - LOW    (>= 0.50) → LOW_CONFIDENCE / MANUAL_REVIEW
  - REJECT (< 0.30)  → REJECT / POOR_QUALITY / DISCARD
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.interfaces.detection_strategy_interface import (
    IDetectionStrategy,
)
from .strategy_configuration import StrategyConfiguration
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


class ConfidenceThresholdStrategy(IDetectionStrategy):
    """
    Production confidence threshold detection strategy plugin.
    """

    def __init__(self, config: Optional[StrategyConfiguration] = None) -> None:
        self._config = config or StrategyConfiguration()
        self._threshold_mgr = ThresholdManager(self._config)
        self._initialized = False

    @property
    def strategy_key(self) -> str:
        return "confidence_threshold"

    @property
    def strategy_version(self) -> str:
        return "1.0.0"

    @property
    def strategy_description(self) -> str:
        return "Baseline per-modality confidence threshold anomaly detection strategy."

    @property
    def requires_feature_vector(self) -> bool:
        return False

    @property
    def requires_model(self) -> bool:
        return False

    def initialize(self, config: Dict[str, Any], model_adapter: Optional[Any] = None) -> None:
        self._initialized = True
        logger.info("[ConfidenceThresholdStrategy] Initialized.")

    def detect(self, context: AnomalyContext) -> bool:
        t0 = time.perf_counter()
        modality = context.modality
        confidence = context.confidence

        level = self._threshold_mgr.evaluate_confidence_level(confidence, modality)
        th = self._threshold_mgr.get_thresholds(modality)

        is_anomaly = level in ("LOW", "REJECT")
        flag = "TRUSTED" if level in ("HIGH", "MEDIUM") else ("LOW_CONFIDENCE" if level == "LOW" else "CRITICAL")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        context.confidence_analysis = {
            "confidence_score": round(confidence, 4),
            "confidence_level": level,
            "confidence_flag": flag,
            "threshold_low": th.low_confidence,
            "threshold_critical": th.reject_threshold,
            "threshold_high": th.high_confidence,
            "threshold_medium": th.medium_confidence,
            "modality": modality,
            "analysis_time_ms": round(elapsed_ms, 3),
        }

        context.record_metric("confidence_eval_time_ms", elapsed_ms)
        return is_anomaly

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self.strategy_key,
            "version": self.strategy_version,
            "description": self.strategy_description,
            "requires_feature_vector": self.requires_feature_vector,
            "requires_model": self.requires_model,
            "is_initialized": self._initialized,
        }


__all__ = ["ConfidenceThresholdStrategy"]
