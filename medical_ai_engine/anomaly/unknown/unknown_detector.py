"""
unknown_detector.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Orchestrates the full unknown disease detection pipeline.

Pipeline
--------
1. FeatureEmbeddingManager  → extract query embedding
2. PrototypeManager         → retrieve known-class prototypes
3. SimilarityEngine         → compute similarity to all prototypes
4. NoveltyEstimator         → compute calibrated novelty score
5. Decision logic           → KNOWN_DISEASE vs UNKNOWN_DISEASE
6. AnomalyResult enrichment → populate known_unknown_status, novelty_score,
                              similarity_score, risk_level, recommended_action, reason
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from ..anomaly_result import AnomalyResult, AnomalyStatus, RecommendedAction, RiskLevel
from .feature_embedding_manager import EmbeddingRecord, FeatureEmbeddingManager
from .novelty_estimator import NoveltyEstimate, NoveltyEstimator
from .prototype_manager import PrototypeManager
from .similarity_engine import SimilarityEngine, SimilarityMetric
from .unknown_configuration import UnknownConfiguration
from .unknown_exceptions import UnknownDetectionError

logger = logging.getLogger(__name__)

# Default set of seed classes used when no model adapter is present
_DEFAULT_KNOWN_CLASSES = [
    "pneumonia", "tuberculosis", "covid19", "lung_cancer",
    "brain_tumor", "alzheimer", "normal", "cardiomegaly",
    "pleural_effusion", "atelectasis",
]


class UnknownDiseaseDetector:
    """
    Orchestrates the Unknown Disease Detection pipeline.

    Determines whether a prediction belongs to a **known** or **unknown** disease
    category by comparing its feature embedding to pre-built class prototypes.

    Parameters
    ----------
    config : UnknownConfiguration
        Thresholds and metric settings.
    known_classes : list[str] or None
        If provided, synthetic prototypes are seeded for these classes.
        Defaults to a standard set of 10 medical disease labels.
    model_adapter : Any or None
        Optional model adapter exposing ``get_features(prediction_result)``.
    """

    def __init__(
        self,
        config: Optional[UnknownConfiguration] = None,
        known_classes: Optional[list] = None,
        model_adapter: Optional[Any] = None,
    ) -> None:
        self._cfg = config or UnknownConfiguration.default()
        self._adapter = model_adapter
        self._lock = threading.Lock()

        # Build the pipeline components
        self._emb_mgr = FeatureEmbeddingManager(embedding_dim=self._cfg.embedding_dim)
        self._proto_mgr = PrototypeManager(
            embedding_manager=self._emb_mgr,
            min_samples=self._cfg.min_samples_per_class,
        )
        self._sim_engine = SimilarityEngine(
            prototype_manager=self._proto_mgr,
            metric=SimilarityMetric.COSINE if self._cfg.use_cosine_similarity else SimilarityMetric.EUCLIDEAN,
        )
        self._novelty_est = NoveltyEstimator()

        # Seed synthetic prototypes for the known class set
        classes = known_classes or _DEFAULT_KNOWN_CLASSES
        self._proto_mgr.seed_synthetic_prototypes(classes, self._cfg.embedding_dim)
        logger.info(
            "UnknownDiseaseDetector initialised (metric=%s, classes=%d)",
            self._sim_engine.metric.value, len(classes),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, prediction_result: Any) -> AnomalyResult:
        """
        Run the unknown disease detection pipeline on *prediction_result*.

        Parameters
        ----------
        prediction_result : Any
            Must expose ``request_id``, ``modality``, ``predicted_disease``,
            ``confidence``, and ``extra`` attributes.

        Returns
        -------
        AnomalyResult
            Enriched with ``known_unknown_status``, ``novelty_score``,
            ``similarity_score``, ``risk_level``, ``recommended_action``, and ``reason``.
        """
        t_start = time.perf_counter()
        try:
            modality = getattr(prediction_result, "modality", "unknown")
            confidence = float(getattr(prediction_result, "confidence", 0.5))

            # 1. Feature extraction
            embedding = self._emb_mgr.extract(prediction_result, self._adapter)

            # 2. Similarity computation
            similarity_result = self._sim_engine.compute(embedding)

            # 3. Novelty estimation
            novelty_estimate = self._novelty_est.estimate(similarity_result, confidence)

            # 4. Decision
            threshold = self._cfg.get_novelty_threshold(modality)
            is_unknown = novelty_estimate.novelty_score > threshold
            sim_below = similarity_result.similarity_score < self._cfg.similarity_threshold

            # Only declare UNKNOWN if BOTH signals agree
            is_unknown = is_unknown and sim_below

            result = self._build_result(
                prediction_result=prediction_result,
                novelty_estimate=novelty_estimate,
                similarity_score=similarity_result.similarity_score,
                nearest_class=similarity_result.nearest_class,
                is_unknown=is_unknown,
                processing_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

            logger.info(
                "Unknown detection [%s]: novelty=%.4f sim=%.4f decision=%s",
                getattr(prediction_result, "request_id", "?"),
                novelty_estimate.novelty_score,
                similarity_result.similarity_score,
                result.known_unknown_status,
            )
            return result

        except UnknownDetectionError:
            raise
        except Exception as exc:
            raise UnknownDetectionError(f"Unknown disease detection failed: {exc}") from exc

    def health_check(self) -> Dict[str, Any]:
        return {
            "component": "UnknownDiseaseDetector",
            "status": "healthy",
            "known_classes": self._proto_mgr.known_classes(),
            "metric": self._sim_engine.metric.value,
            "embedding_dim": self._cfg.embedding_dim,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        prediction_result: Any,
        novelty_estimate: NoveltyEstimate,
        similarity_score: float,
        nearest_class: str,
        is_unknown: bool,
        processing_time_ms: float,
    ) -> AnomalyResult:
        """Construct and return the AnomalyResult."""
        if is_unknown:
            status = AnomalyStatus.UNKNOWN_DISEASE
            known_unknown = "UNKNOWN"
            risk = RiskLevel.HIGH
            action = RecommendedAction.ESCALATE
            reason = (
                f"Unknown disease detected: novelty={novelty_estimate.novelty_score:.3f}, "
                f"similarity={similarity_score:.3f} (threshold={self._cfg.similarity_threshold}). "
                f"Nearest known class: '{nearest_class}'."
            )
        else:
            status = AnomalyStatus.KNOWN_DISEASE
            known_unknown = "KNOWN"
            risk = RiskLevel.LOW
            action = RecommendedAction.PROCEED
            reason = (
                f"Known disease identified: novelty={novelty_estimate.novelty_score:.3f}, "
                f"similarity={similarity_score:.3f}. "
                f"Matched class: '{nearest_class}'."
            )

        unknown_analysis = {
            "is_unknown": is_unknown,
            "novelty_score": novelty_estimate.novelty_score,
            "similarity_score": similarity_score,
            "nearest_class": nearest_class,
            "similarity_threshold": self._cfg.similarity_threshold,
        }

        return AnomalyResult(
            request_id=getattr(prediction_result, "request_id", "unknown"),
            session_id=getattr(prediction_result, "session_id", ""),
            status=status,
            risk_level=risk,
            known_unknown_status=known_unknown,
            confidence_score=float(getattr(prediction_result, "confidence", 0.0)),
            risk_score=novelty_estimate.novelty_score,
            recommended_action=action,
            reason=reason,
            detection_method="unknown_disease_detector",
            processing_time_ms=processing_time_ms,
            unknown_disease_analysis=unknown_analysis,
            metadata={
                "novelty_score": novelty_estimate.novelty_score,
                "similarity_score": similarity_score,
                "nearest_class": nearest_class,
            },
        )


__all__ = ["UnknownDiseaseDetector"]
