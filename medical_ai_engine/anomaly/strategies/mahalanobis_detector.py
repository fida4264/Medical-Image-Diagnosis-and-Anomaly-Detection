"""
mahalanobis_detector.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Mahalanobis Distance OOD Detection Strategy implementing IDetectionStrategy.

Strategy Key
------------
"mahalanobis"

Functionality
-------------
1. Extracts penultimate features via FeatureExtractor.
2. Computes Mahalanobis distance D_M(z, mu_c) to class mean prototypes via DistanceCalculator.
3. Compares minimum distance against per-modality OOD threshold.
4. Returns True if distance > threshold (Out-of-Distribution).
5. Populates context.ood_analysis with distance_score, nearest_class, and is_ood decision.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.interfaces.detection_strategy_interface import (
    IDetectionStrategy,
)
from .distance_calculator import DistanceCalculator
from .feature_extractor import FeatureExtractor
from .feature_statistics import FeatureStatistics
from .strategy_configuration import StrategyConfiguration
from .threshold_manager import ThresholdManager

logger = logging.getLogger(__name__)


class MahalanobisOODStrategy(IDetectionStrategy):
    """
    Production Mahalanobis Distance OOD Detection Strategy plugin.
    """

    def __init__(self, config: Optional[StrategyConfiguration] = None) -> None:
        self._config = config or StrategyConfiguration()
        self._feature_extractor = FeatureExtractor()
        self._statistics = FeatureStatistics(regularization=self._config.covariance_regularization)
        self._threshold_mgr = ThresholdManager(self._config)
        self._model_adapter: Optional[Any] = None
        self._initialized = False

    @property
    def strategy_key(self) -> str:
        return "mahalanobis"

    @property
    def strategy_version(self) -> str:
        return "1.0.0"

    @property
    def strategy_description(self) -> str:
        return "Mahalanobis distance Out-of-Distribution (OOD) detection strategy."

    @property
    def requires_feature_vector(self) -> bool:
        return True

    @property
    def requires_model(self) -> bool:
        return False  # Operates on features or stubs cleanly

    def initialize(self, config: Dict[str, Any], model_adapter: Optional[Any] = None) -> None:
        self._model_adapter = model_adapter
        # Fit synthetic class statistics stubs for common platform classes if not pre-fitted
        if not self._statistics.is_fitted:
            default_classes = ["glioma", "meningioma", "pituitary", "pneumonia", "normal", "melanoma", "benign"]
            self._statistics.fit_synthetic_defaults(default_classes, feature_dim=512)

        self._initialized = True
        logger.info("[MahalanobisOODStrategy] Initialized.")

    def detect(self, context: AnomalyContext) -> bool:
        t0 = time.perf_counter()
        modality = context.modality

        # 1. Feature Extraction
        t_feat = time.perf_counter()
        feature_vec = self._feature_extractor.extract_features(
            model_adapter=self._model_adapter,
            image_input=None,
            context_features=context.feature_vector,
        )
        feat_time_ms = (time.perf_counter() - t_feat) * 1000.0

        # Store extracted features on context for downstream strategies
        context.feature_vector = feature_vec

        # 2. Distance Computation
        t_dist = time.perf_counter()
        dist, nearest_cls = DistanceCalculator.compute_mahalanobis(
            feature_vec=feature_vec,
            class_means=self._statistics.class_means,
            precision_matrix=self._statistics.precision_matrix,
        )
        dist_time_ms = (time.perf_counter() - t_dist) * 1000.0

        context.feature_distance = dist

        # 3. Threshold Evaluation
        th = self._threshold_mgr.get_thresholds(modality)
        is_ood = dist > th.mahalanobis_ood_threshold

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Populate Context Analysis
        context.ood_analysis = {
            "is_ood": is_ood,
            "ood_score": round(dist, 4),
            "distance": round(dist, 4),
            "nearest_class": nearest_cls,
            "threshold": th.mahalanobis_ood_threshold,
            "modality": modality,
            "strategy_used": self.strategy_key,
            "feature_extraction_ms": round(feat_time_ms, 3),
            "distance_computation_ms": round(dist_time_ms, 3),
            "total_ms": round(elapsed_ms, 3),
        }

        context.record_metric("feature_extraction_time_ms", feat_time_ms)
        context.record_metric("distance_computation_time_ms", dist_time_ms)
        context.record_metric("mahalanobis_total_time_ms", elapsed_ms)

        logger.debug(
            "[MahalanobisOODStrategy] Modality '%s': distance=%.4f (threshold=%.2f) -> is_ood=%s",
            modality, dist, th.mahalanobis_ood_threshold, is_ood
        )

        return is_ood

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self.strategy_key,
            "version": self.strategy_version,
            "description": self.strategy_description,
            "requires_feature_vector": self.requires_feature_vector,
            "requires_model": self.requires_model,
            "is_initialized": self._initialized,
            "is_statistics_fitted": self._statistics.is_fitted,
        }


__all__ = ["MahalanobisOODStrategy"]
