"""
threshold_manager.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Threshold lookup, tuning, and per-modality manager.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .strategy_configuration import ModalityThresholds, StrategyConfiguration
from .strategy_exceptions import ThresholdManagerError

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages dynamic lookup and clinical threshold tuning across modalities.
    """

    def __init__(self, config: Optional[StrategyConfiguration] = None) -> None:
        self.config = config or StrategyConfiguration()

    def get_thresholds(self, modality: str) -> ModalityThresholds:
        return self.config.get_thresholds_for_modality(modality)

    def update_modality_thresholds(self, modality: str, new_thresholds: ModalityThresholds) -> None:
        self.config.modality_thresholds[modality] = new_thresholds
        logger.info("[ThresholdManager] Updated thresholds for modality '%s'.", modality)

    def evaluate_confidence_level(self, confidence: float, modality: str) -> str:
        """
        Evaluate confidence into: HIGH, MEDIUM, LOW, or REJECT.
        """
        th = self.get_thresholds(modality)
        if confidence >= th.high_confidence:
            return "HIGH"
        if confidence >= th.medium_confidence:
            return "MEDIUM"
        if confidence >= th.low_confidence:
            return "LOW"
        return "REJECT"

    def is_mahalanobis_ood(self, distance: float, modality: str) -> bool:
        th = self.get_thresholds(modality)
        return distance > th.mahalanobis_ood_threshold


__all__ = ["ThresholdManager"]
