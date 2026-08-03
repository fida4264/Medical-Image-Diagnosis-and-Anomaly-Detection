"""
novelty_estimator.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Estimates novelty score from similarity and prediction context.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict

from .similarity_engine import SimilarityResult
from .unknown_exceptions import NoveltyEstimatorError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoveltyEstimate:
    """
    Novelty estimation result.

    Attributes
    ----------
    novelty_score : float
        A calibrated score in [0.0, 1.0].
        0.0 → clearly known;  1.0 → maximally novel.
    similarity_score : float
        Raw similarity to nearest prototype (from SimilarityEngine).
    nearest_class : str
        Nearest known-class prototype.
    confidence : float
        Prediction confidence forwarded from context.
    """
    novelty_score: float
    similarity_score: float
    nearest_class: str
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "novelty_score": round(self.novelty_score, 6),
            "similarity_score": round(self.similarity_score, 6),
            "nearest_class": self.nearest_class,
            "confidence": round(self.confidence, 6),
        }


class NoveltyEstimator:
    """
    Combines prototype-similarity and prediction-confidence into a single
    calibrated novelty score.

    Algorithm
    ---------
    Raw novelty = 1 − similarity_score
    Confidence penalty = 1 − confidence   (high confidence → low penalty)
    Combined = α * raw_novelty + β * confidence_penalty
    Calibrated = sigmoid-like smoothing to keep score in [0, 1]

    Default weights: α=0.70, β=0.30 (similarity dominates).
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.30,
    ) -> None:
        if not math.isclose(alpha + beta, 1.0, rel_tol=1e-6):
            raise NoveltyEstimatorError(
                f"alpha + beta must equal 1.0; got alpha={alpha}, beta={beta}"
            )
        self._alpha = alpha
        self._beta = beta

    def estimate(
        self,
        similarity_result: SimilarityResult,
        confidence: float,
    ) -> NoveltyEstimate:
        """
        Estimate the novelty score from similarity and prediction confidence.

        Parameters
        ----------
        similarity_result : SimilarityResult
            Output from SimilarityEngine.compute().
        confidence : float
            Predicted disease confidence in [0, 1].

        Returns
        -------
        NoveltyEstimate
        """
        if not 0.0 <= confidence <= 1.0:
            raise NoveltyEstimatorError(f"confidence must be in [0,1]; got {confidence}")

        sim = float(similarity_result.similarity_score)
        raw_novelty = max(0.0, 1.0 - sim)
        conf_penalty = max(0.0, 1.0 - float(confidence))

        combined = self._alpha * raw_novelty + self._beta * conf_penalty
        novelty_score = self._calibrate(combined)

        logger.debug(
            "Novelty: sim=%.4f raw_novelty=%.4f conf=%.4f penalty=%.4f combined=%.4f final=%.4f",
            sim, raw_novelty, confidence, conf_penalty, combined, novelty_score,
        )

        return NoveltyEstimate(
            novelty_score=novelty_score,
            similarity_score=sim,
            nearest_class=similarity_result.nearest_class,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _calibrate(x: float) -> float:
        """Apply a sigmoid-like smoothing that maps (0,1) → (0,1)."""
        # tanh-based normalisation: tanh(2*x) saturates at extremes more gracefully
        return float(math.tanh(2.0 * x))

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta


__all__ = ["NoveltyEstimator", "NoveltyEstimate"]
