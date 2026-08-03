"""
feature_extractor.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Feature extraction module from PyTorch/Torchvision classification models.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Optional

from .strategy_exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts high-dimensional penultimate feature vectors from classification models.

    Supports:
      1. Direct feature vector attached to EngineContext / AnomalyContext.
      2. PyTorch model hook feature extraction (penultimate layer).
      3. Fallback deterministic synthetic feature generator when model is unavailable or in stub mode.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        self.feature_dim = feature_dim

    def extract_features(self, model_adapter: Optional[Any], image_input: Optional[Any], context_features: Optional[Any] = None) -> np.ndarray:
        """
        Extract a 1D float32 feature vector [D].

        Parameters
        ----------
        model_adapter : Optional[Any]
            Trained model adapter instance.
        image_input : Optional[Any]
            Input image array or tensor.
        context_features : Optional[Any]
            Pre-extracted features already present on context.

        Returns
        -------
        np.ndarray
            1D float32 numpy array of dimension (D,).
        """
        # Case 1: Pre-extracted features provided in context
        if context_features is not None:
            arr = np.asarray(context_features, dtype=np.float32).ravel()
            if arr.size > 0:
                return arr

        # Case 2: PyTorch model forward pass feature extraction via adapter
        if model_adapter is not None and hasattr(model_adapter, "extract_penultimate_features"):
            try:
                feats = model_adapter.extract_penultimate_features(image_input)
                return np.asarray(feats, dtype=np.float32).ravel()
            except Exception as exc:
                logger.warning("[FeatureExtractor] Model feature extraction failed: %s; falling back to stub.", exc)

        # Case 3: Synthetic deterministic feature stub for testing/stubs
        seed = 42
        if image_input is not None:
            try:
                seed = int(np.sum(np.asarray(image_input))) % 10000
            except Exception:
                seed = 42

        rng = np.random.RandomState(seed)
        return rng.randn(self.feature_dim).astype(np.float32)


__all__ = ["FeatureExtractor"]
