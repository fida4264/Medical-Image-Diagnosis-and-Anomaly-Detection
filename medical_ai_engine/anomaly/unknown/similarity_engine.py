"""
similarity_engine.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Similarity computation between query embedding and class prototypes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .prototype_manager import ClassPrototype, PrototypeManager
from .unknown_exceptions import SimilarityEngineError

logger = logging.getLogger(__name__)


class SimilarityMetric(str, Enum):
    """Supported similarity / distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass(frozen=True)
class SimilarityResult:
    """
    Result of comparing a query embedding against class prototypes.

    Attributes
    ----------
    nearest_class : str
        Class label of the most similar prototype.
    similarity_score : float
        Similarity to nearest prototype (0–1 for cosine; lower-is-better for Euclidean).
    metric : SimilarityMetric
        Metric used for computation.
    all_scores : Dict[str, float]
        Scores for every known class prototype.
    """
    nearest_class: str
    similarity_score: float
    metric: SimilarityMetric
    all_scores: Dict[str, float]

    def as_dict(self) -> dict:
        return {
            "nearest_class": self.nearest_class,
            "similarity_score": round(self.similarity_score, 6),
            "metric": self.metric.value,
        }


class SimilarityEngine:
    """
    Computes similarity between a query embedding and all registered class prototypes.

    Supports
    --------
    - **Cosine similarity**: cos(θ) = (q · μ) / (‖q‖ · ‖μ‖)
      Range: [−1, 1] → normalised to [0, 1]. Higher = more similar.
    - **Euclidean distance**: d = ‖q − μ‖₂
      Converted to similarity = 1/(1+d). Higher = more similar.
    - **Dot product**: q · μ (vectors assumed unit-normalised).
    """

    def __init__(
        self,
        prototype_manager: PrototypeManager,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self._pm = prototype_manager
        self._metric = metric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, query: np.ndarray) -> SimilarityResult:
        """
        Compute similarity scores against all known class prototypes.

        Parameters
        ----------
        query : np.ndarray
            Query feature embedding of shape (D,).

        Returns
        -------
        SimilarityResult
        """
        prototypes = self._pm.all_prototypes()
        if not prototypes:
            raise SimilarityEngineError(
                "No class prototypes available; call PrototypeManager.fit() or "
                "seed_synthetic_prototypes() first."
            )
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1:
            raise SimilarityEngineError(f"query must be 1-D; got shape {query.shape}")

        scores: Dict[str, float] = {}
        for proto in prototypes:
            scores[proto.class_label] = self._score(query, proto)

        nearest = max(scores, key=lambda k: scores[k])
        return SimilarityResult(
            nearest_class=nearest,
            similarity_score=scores[nearest],
            metric=self._metric,
            all_scores=scores,
        )

    # ------------------------------------------------------------------
    # Metric Helpers
    # ------------------------------------------------------------------

    def _score(self, query: np.ndarray, proto: ClassPrototype) -> float:
        if self._metric == SimilarityMetric.COSINE:
            return self._cosine(query, proto.centroid)
        if self._metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_sim(query, proto.centroid)
        if self._metric == SimilarityMetric.DOT_PRODUCT:
            return float(np.dot(query, proto.centroid))
        raise SimilarityEngineError(f"Unknown metric: {self._metric}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        raw = float(np.dot(a, b)) / denom
        # Normalise from [−1, 1] → [0, 1]
        return (raw + 1.0) / 2.0

    @staticmethod
    def _euclidean_sim(a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    def top_k(self, query: np.ndarray, k: int = 3) -> List[SimilarityResult]:
        """Return top-k most similar prototypes as individual SimilarityResult objects."""
        full = self.compute(query)
        ranked = sorted(full.all_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            SimilarityResult(
                nearest_class=label,
                similarity_score=score,
                metric=self._metric,
                all_scores=full.all_scores,
            )
            for label, score in ranked
        ]

    @property
    def metric(self) -> SimilarityMetric:
        return self._metric


__all__ = ["SimilarityEngine", "SimilarityMetric", "SimilarityResult"]
