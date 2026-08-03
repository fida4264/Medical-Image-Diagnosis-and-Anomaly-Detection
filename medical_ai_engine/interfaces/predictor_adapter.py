"""
Medical AI Engine — Predictor Adapter Interface

Abstract Base Class defining the mandatory interface for all disease module prediction adapters.
No adapter-specific methods may be called directly by the engine.

Required methods:
  - initialize()
  - load_model()
  - predict()
  - health_check()
  - get_metadata()
  - shutdown()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from PIL import Image


class IPredictorAdapter(ABC):
    """
    Abstract contract for disease prediction adapters wrapping Modules 7–13.
    Unified interface guarantees zero model-specific knowledge in the Engine Core.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the lowercase target modality string (e.g. 'skin', 'retina')."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the underlying model weights are loaded in memory."""
        pass

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the adapter configuration and paths.
        Does not load heavy model weights into memory until load_model() is called.
        """
        pass

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None, label_map_path: Optional[str] = None) -> None:
        """
        Load model weights and class label mapping into memory.
        Must be thread-safe.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Execute prediction on a PIL Image.

        Returns:
            Dictionary conforming to standardized prediction schema.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Return health status dictionary of the adapter and loaded model.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata dictionary detailing modality, supported classes,
        model version, framework, creation date, status, health, and export formats.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Unload model weights and release VRAM/RAM resources.
        """
        pass
