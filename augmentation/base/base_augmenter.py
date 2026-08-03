"""
augmentation/base/base_augmenter.py
=====================================
Abstract Base Class for all modality-specific augmenters.

Every policy (BrainMRI, BrainCT, ChestXRay, etc.) inherits from this
class and implements `build_pipeline()`.

Design Principles
-----------------
- Each augmenter receives a BGR uint8 numpy array and returns:
    (augmented_image: np.ndarray, operations_applied: List[str])
- Reproducible: a seed per augmentation call controls all randomness.
- Stateless: the same instance can be called concurrently (thread-safe).
- Configuration is injected at construction time from the YAML config.

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseAugmenter(ABC):
    """
    Abstract base class for modality-specific image augmenters.

    Parameters
    ----------
    config : dict
        Full augmentation configuration dict loaded from
        ``configs/augmentation.yaml``. The modality-specific section
        is accessible via ``self.modality_config``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._common = config.get("common", {})
        self._modality_config = config.get("modalities", {}).get(
            self.modality, {}
        )
        self._pipeline: A.Compose = self.build_pipeline(config)
        logger.debug(
            "Initialised %s (modality=%s)",
            self.__class__.__name__,
            self.modality,
        )

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the canonical modality string (e.g., 'brain_mri')."""

    @abstractmethod
    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build and return the albumentations pipeline for this modality.

        Parameters
        ----------
        config : dict
            Full augmentation configuration.

        Returns
        -------
        albumentations.Compose
            The complete augmentation pipeline.
        """

    # ------------------------------------------------------------------
    # Concrete augment method — shared by all subclasses
    # ------------------------------------------------------------------

    def augment(
        self,
        image: np.ndarray,
        seed: int = 42,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply the augmentation pipeline to one image.

        Parameters
        ----------
        image : np.ndarray
            Input image as a BGR uint8 numpy array (H, W, 3).
        seed : int
            Random seed for this specific augmentation call.

        Returns
        -------
        augmented : np.ndarray
            Augmented image as a BGR uint8 numpy array, same spatial size.
        operations : list of str
            Human-readable names of operations applied.
        """
        # Convert BGR → RGB for albumentations
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Seed the pipeline using the official albumentations ≥ 1.4 API.
        # This sets the internal numpy Generator used by all transforms,
        # guaranteeing reproducibility independent of the global random state.
        self._pipeline.set_random_seed(seed)

        # Apply pipeline
        result = self._pipeline(image=rgb)
        augmented_rgb = result["image"]

        # Convert back to BGR
        augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

        ops = self._extract_ops()
        logger.debug(
            "[%s] augment seed=%d | ops=%s", self.modality, seed, ops
        )
        return augmented_bgr, ops

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _extract_ops(self) -> List[str]:
        """Return a list of transform names from the pipeline."""
        ops = []
        for t in self._pipeline.transforms:
            name = type(t).__name__
            ops.append(name)
        return ops

    def cfg(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the modality config, falling back to
        the common config, then to ``default``.
        """
        return self._modality_config.get(
            key, self._common.get(key, default)
        )

    def _p(self, section: str, default: float = 0.5) -> float:
        """Get probability for a named transform section."""
        mod_val = self._modality_config.get(section, {})
        com_val = self._common.get(section, {})
        if isinstance(mod_val, dict) and "probability" in mod_val:
            return float(mod_val["probability"])
        if isinstance(com_val, dict) and "probability" in com_val:
            return float(com_val["probability"])
        return default

    def _enabled(self, section: str, default: bool = True) -> bool:
        """Check if a named transform section is enabled."""
        mod_val = self._modality_config.get(section, {})
        com_val = self._common.get(section, {})
        if isinstance(mod_val, dict) and "enabled" in mod_val:
            return bool(mod_val["enabled"])
        if isinstance(com_val, dict) and "enabled" in com_val:
            return bool(com_val["enabled"])
        return default

    def _get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific key from a named config section."""
        mod_val = self._modality_config.get(section, {})
        com_val = self._common.get(section, {})
        if isinstance(mod_val, dict) and key in mod_val:
            return mod_val[key]
        if isinstance(com_val, dict) and key in com_val:
            return com_val[key]
        return default

    @property
    def modality_config(self) -> Dict[str, Any]:
        """Full modality-specific configuration section."""
        return self._modality_config

    @property
    def common_config(self) -> Dict[str, Any]:
        """Common configuration section."""
        return self._common

    def __repr__(self) -> str:
        n = len(self._pipeline.transforms)
        return (
            f"{self.__class__.__name__}("
            f"modality={self.modality!r}, "
            f"transforms={n})"
        )
