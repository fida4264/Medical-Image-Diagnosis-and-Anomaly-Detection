"""
preprocessing/factory/preprocessor_factory.py
================================================
Factory for creating the correct modality-specific preprocessor.

Usage
-----
    factory = PreprocessorFactory(config)
    preprocessor = factory.get("brain_mri")
    processed, ops = preprocessor.preprocess(image)

Design
------
- Registry-based: maps modality strings to preprocessor classes.
- Thread-safe: creates a new preprocessor instance per call.
- Validates modality against the registry before instantiation.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.pipelines.brain_mri_preprocessor import BrainMRIPreprocessor
from preprocessing.pipelines.brain_ct_preprocessor import BrainCTPreprocessor
from preprocessing.pipelines.chest_xray_preprocessor import ChestXRayPreprocessor
from preprocessing.pipelines.retina_preprocessor import RetinaPreprocessor
from preprocessing.pipelines.skin_preprocessor import SkinPreprocessor
from preprocessing.pipelines.ultrasound_preprocessor import UltrasoundPreprocessor
from preprocessing.pipelines.histopathology_preprocessor import HistopathologyPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BasePreprocessor]] = {
    "brain_mri":      BrainMRIPreprocessor,
    "brain_ct":       BrainCTPreprocessor,
    "chest_xray":     ChestXRayPreprocessor,
    "retina":         RetinaPreprocessor,
    "skin":           SkinPreprocessor,
    "ultrasound":     UltrasoundPreprocessor,
    "histopathology": HistopathologyPreprocessor,
}


class PreprocessorFactory:
    """
    Creates modality-specific preprocessor instances.

    Parameters
    ----------
    config : dict
        Full preprocessing configuration (loaded from preprocessing.yaml).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def get(self, modality: str) -> BasePreprocessor:
        """
        Return an instantiated preprocessor for the given modality.

        Parameters
        ----------
        modality : str
            Modality string from master_metadata.csv
            (e.g., 'brain_mri', 'chest_xray').

        Returns
        -------
        BasePreprocessor
            Configured preprocessor for that modality.

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
        return f"PreprocessorFactory(modalities={self.supported_modalities()})"
