"""
feature_embedding_manager.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Manages feature embedding extraction and storage.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .unknown_exceptions import EmbeddingManagerError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """A single stored feature embedding linked to a class label."""
    vector: np.ndarray
    class_label: str
    modality: str
    sample_id: str
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "class_label": self.class_label,
            "modality": self.modality,
            "sample_id": self.sample_id,
            "embedding_dim": len(self.vector),
            "timestamp": self.timestamp,
        }


class FeatureEmbeddingManager:
    """
    Manages feature embedding extraction and storage for the Unknown Disease
    Detection subsystem.

    Responsibilities
    ----------------
    - Extract normalised feature embeddings from prediction context or model adapters.
    - Store embeddings per class label and modality for prototype construction.
    - Support deterministic synthetic generation for test environments.
    - Provide thread-safe access to stored embeddings.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise EmbeddingManagerError(f"embedding_dim must be > 0; got {embedding_dim}")
        self._dim = embedding_dim
        # { class_label -> List[EmbeddingRecord] }
        self._store: Dict[str, List[EmbeddingRecord]] = {}
        self._lock = __import__("threading").Lock()
        logger.info("FeatureEmbeddingManager initialised (dim=%d)", embedding_dim)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        prediction_result: Any,
        model_adapter: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Extract a feature embedding from *prediction_result*.

        Strategy (in priority order):
        1. If the result carries a pre-computed ``feature_vector``, use it.
        2. If a *model_adapter* with ``get_features()`` is provided, call it.
        3. Fall back to a deterministic synthetic embedding keyed by request_id.

        Returns
        -------
        np.ndarray
            Normalised embedding vector of shape ``(embedding_dim,)``.
        """
        vector = self._try_extract_from_result(prediction_result)
        if vector is None and model_adapter is not None:
            vector = self._try_extract_from_adapter(prediction_result, model_adapter)
        if vector is None:
            vector = self._synthetic_embedding(prediction_result)

        vector = self._normalise(vector)
        logger.debug("Extracted embedding (dim=%d, norm=%.4f)", len(vector), float(np.linalg.norm(vector)))
        return vector

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, record: EmbeddingRecord) -> None:
        """Store a labelled embedding record."""
        with self._lock:
            self._store.setdefault(record.class_label, []).append(record)

    def get_embeddings(self, class_label: str) -> List[EmbeddingRecord]:
        """Return all stored records for *class_label*."""
        with self._lock:
            return list(self._store.get(class_label, []))

    def known_classes(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def class_sample_count(self, class_label: str) -> int:
        return len(self.get_embeddings(class_label))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_extract_from_result(self, result: Any) -> Optional[np.ndarray]:
        vec = getattr(result, "feature_vector", None)
        if vec is None:
            extra = getattr(result, "extra", {}) or {}
            vec = extra.get("feature_vector")
        if vec is not None:
            return np.asarray(vec, dtype=np.float32)
        return None

    def _try_extract_from_adapter(self, result: Any, adapter: Any) -> Optional[np.ndarray]:
        try:
            vec = adapter.get_features(result)
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:
            logger.warning("Model adapter feature extraction failed: %s", exc)
            return None

    def _synthetic_embedding(self, result: Any) -> np.ndarray:
        """
        Deterministic synthetic embedding derived from request_id and
        predicted_disease so the same prediction always produces the same vector.
        """
        seed_str = (
            str(getattr(result, "request_id", ""))
            + str(getattr(result, "predicted_disease", ""))
            + str(getattr(result, "modality", ""))
        )
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim).astype(np.float32)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return vec
        return vec / norm

    @property
    def embedding_dim(self) -> int:
        return self._dim


__all__ = ["FeatureEmbeddingManager", "EmbeddingRecord"]
