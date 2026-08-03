"""
augmentation/factory/augmentation_factory.py
==============================================
Factory for creating the correct modality-specific augmenter.

Usage
-----
    factory = AugmentationFactory(config)
    augmenter = factory.get("brain_mri")
    augmented, ops = augmenter.augment(image, seed=42)

Design
------
- Registry-based: maps modality strings to augmenter classes.
- Thread-safe: creates a new augmenter instance per call.
- Validates modality against the registry before instantiation.

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from augmentation.base.base_augmenter import BaseAugmenter
from augmentation.policies.brain_mri_policy import BrainMRIAugmenter
from augmentation.policies.brain_ct_policy import BrainCTAugmenter
from augmentation.policies.chest_xray_policy import ChestXRayAugmenter
from augmentation.policies.retina_policy import RetinaAugmenter
from augmentation.policies.skin_policy import SkinAugmenter
from augmentation.policies.ultrasound_policy import UltrasoundAugmenter
from augmentation.policies.histopathology_policy import HistopathologyAugmenter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseAugmenter]] = {
    "brain_mri":      BrainMRIAugmenter,
    "brain_ct":       BrainCTAugmenter,
    "chest_xray":     ChestXRayAugmenter,
    "retina":         RetinaAugmenter,
    "skin":           SkinAugmenter,
    "ultrasound":     UltrasoundAugmenter,
    "histopathology": HistopathologyAugmenter,
}


class AugmentationFactory:
    """
    Creates modality-specific augmenter instances.

    Parameters
    ----------
    config : dict
        Full augmentation configuration (loaded from augmentation.yaml).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def get(self, modality: str) -> BaseAugmenter:
        """
        Return an instantiated augmenter for the given modality.

        Parameters
        ----------
        modality : str
            Modality string (e.g., 'brain_mri', 'chest_xray').

        Returns
        -------
        BaseAugmenter
            Configured augmenter for that modality.

        Raises
        ------
        ValueError
            If the modality is not registered.
        """
        key = modality.strip().lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown modality: '{modality}'. "
                f"Supported: {sorted(_REGISTRY.keys())}"
            )
        cls = _REGISTRY[key]
        instance = cls(self._config)
        logger.debug("Created %s for modality '%s'", cls.__name__, modality)
        return instance

    @staticmethod
    def supported_modalities() -> List[str]:
        """Return a sorted list of all supported modality strings."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def is_supported(modality: str) -> bool:
        """Return True if the modality is in the registry."""
        return modality.strip().lower() in _REGISTRY

    def __repr__(self) -> str:
        return f"AugmentationFactory(modalities={self.supported_modalities()})"
