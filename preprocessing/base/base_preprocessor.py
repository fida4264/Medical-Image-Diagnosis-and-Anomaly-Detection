"""
preprocessing/base/base_preprocessor.py
=========================================
Abstract Base Class for all modality-specific image preprocessors.

Every pipeline (BrainMRI, BrainCT, ChestXRay, etc.) inherits from this
class and implements the `preprocess` method.

Design Principles
-----------------
- Each preprocessor receives a BGR uint8 numpy array and returns:
    (processed_image: np.ndarray, operations_applied: List[str])
- The preprocessor is stateless: the same instance can be called on
  multiple images concurrently (thread-safe).
- Configuration is injected at construction time from the YAML config.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for modality-specific image preprocessors.

    Parameters
    ----------
    config : dict
        Full preprocessing configuration dict loaded from
        ``configs/preprocessing.yaml``.  The modality-specific section
        is accessible via ``self.modality_config``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._general = config.get("general", {})
        self._modality_config = config.get("modalities", {}).get(
            self.modality, {}
        )
        logger.debug(
            "Initialised %s with target_size=%s",
            self.__class__.__name__,
            self.target_size,
        )

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the canonical modality string (e.g., 'brain_mri')."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Apply the full modality-specific preprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input image as a BGR uint8 numpy array of any size.

        Returns
        -------
        processed : np.ndarray
            Processed image as a BGR uint8 numpy array of shape
            (target_h, target_w, 3).
        operations : list of str
            Human-readable names of every operation applied, in order.
            Example: ['grayscale', 'clahe', 'gaussian_denoise', 'resize']
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    @property
    def target_size(self) -> Tuple[int, int]:
        """Return (height, width) target size for this modality."""
        size = self._modality_config.get(
            "target_size", self._general.get("target_size", [224, 224])
        )
        return (int(size[0]), int(size[1]))

    @property
    def output_channels(self) -> int:
        """Return the required output channel count (1 or 3)."""
        return int(self._modality_config.get("output_channels", 3))

    @property
    def modality_config(self) -> Dict[str, Any]:
        """Full modality-specific configuration section."""
        return self._modality_config

    @property
    def general_config(self) -> Dict[str, Any]:
        """General configuration section."""
        return self._general

    def cfg(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the modality config, falling back to
        the general config, then to ``default``.

        Parameters
        ----------
        key : str
            Configuration key to look up.
        default : any
            Value to return if key is not found in either section.
        """
        return self._modality_config.get(
            key, self._general.get(key, default)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"modality={self.modality!r}, "
            f"target_size={self.target_size})"
        )
