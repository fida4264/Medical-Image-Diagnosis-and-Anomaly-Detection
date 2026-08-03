"""
prototype_manager.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Builds and manages class prototypes from stored embeddings.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .feature_embedding_manager import FeatureEmbeddingManager
from .unknown_exceptions import PrototypeManagerError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassPrototype:
    """Immutable centroid prototype for a single disease class."""
    class_label: str
    centroid: np.ndarray          # mean embedding; shape (D,)
    sample_count: int
    std_dev: float                # mean intra-class std-dev (scalar)

    def as_dict(self) -> dict:
        return {
            "class_label": self.class_label,
            "embedding_dim": len(self.centroid),
            "sample_count": self.sample_count,
            "std_dev": round(self.std_dev, 6),
        }


class PrototypeManager:
    """
    Builds and caches class prototypes (mean centroids) from labelled embeddings.

    The centroid for class *c* is:
        μ_c = (1/N_c) Σ z_i,  z_i ∈ in-distribution samples for c

    Prototypes are rebuilt lazily when new embeddings are stored.

    Thread-Safety
    -------------
    All public methods are protected by a reentrant lock.
    """

    def __init__(
        self,
        embedding_manager: FeatureEmbeddingManager,
        min_samples: int = 5,
    ) -> None:
        if min_samples < 1:
            raise PrototypeManagerError(f"min_samples must be >= 1; got {min_samples}")
        self._emb_mgr = embedding_manager
        self._min_samples = min_samples
        self._prototypes: Dict[str, ClassPrototype] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, class_label: Optional[str] = None) -> None:
        """
        Build / rebuild prototypes.

        Parameters
        ----------
        class_label : Optional[str]
            If given, rebuild only that class. If None, rebuild all known classes.
        """
        targets = [class_label] if class_label else self._emb_mgr.known_classes()
        for label in targets:
            self._build_prototype(label)

    def get_prototype(self, class_label: str) -> Optional[ClassPrototype]:
        with self._lock:
            return self._prototypes.get(class_label)

    def all_prototypes(self) -> List[ClassPrototype]:
        with self._lock:
            return list(self._prototypes.values())

    def known_classes(self) -> List[str]:
        with self._lock:
            return list(self._prototypes.keys())

    def has_prototype(self, class_label: str) -> bool:
        with self._lock:
            return class_label in self._prototypes

    def seed_synthetic_prototypes(self, class_labels: List[str], dim: int) -> None:
        """
        Seed deterministic synthetic prototypes for known classes.

        Each class gets a unique random centroid derived from its label name.
        Useful for unit tests and environments without a trained model.
        """
        with self._lock:
            for i, label in enumerate(class_labels):
                rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
                centroid = rng.standard_normal(dim).astype(np.float32)
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-9:
                    centroid = centroid / norm
                self._prototypes[label] = ClassPrototype(
                    class_label=label,
                    centroid=centroid,
                    sample_count=50,   # synthetic marker
                    std_dev=0.10,
                )
                logger.debug("Seeded synthetic prototype for '%s'", label)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prototype(self, label: str) -> None:
        records = self._emb_mgr.get_embeddings(label)
        if len(records) < self._min_samples:
            logger.debug(
                "Class '%s': only %d samples, need %d — skipping prototype.",
                label, len(records), self._min_samples,
            )
            return
        vectors = np.stack([r.vector for r in records], axis=0)  # (N, D)
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-9:
            centroid = centroid / norm
        std_dev = float(np.std(np.linalg.norm(vectors - centroid, axis=1)))
        with self._lock:
            self._prototypes[label] = ClassPrototype(
                class_label=label,
                centroid=centroid,
                sample_count=len(records),
                std_dev=std_dev,
            )
        logger.info("Prototype built: class='%s' samples=%d std=%.4f", label, len(records), std_dev)


__all__ = ["PrototypeManager", "ClassPrototype"]
