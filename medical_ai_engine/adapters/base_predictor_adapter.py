"""
Medical AI Engine — Base Predictor Adapter

Base class for all disease module prediction adapters.
Implements IPredictorAdapter interface.

Mandatory methods:
  - initialize()
  - load_model()
  - predict()
  - health_check()
  - get_metadata()
  - shutdown()
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from PIL import Image

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter


class BasePredictorAdapter(IPredictorAdapter):
    """
    Abstract base class for all 7 disease module adapters.
    Provides common initialization, status tracking, and default metadata behavior.
    """

    def __init__(self) -> None:
        self._is_initialized: bool = False
        self._is_loaded: bool = False
        self._checkpoint_path: Optional[str] = None
        self._label_map_path: Optional[str] = None
        self._predictor: Optional[Any] = None
        self._config: Dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded into memory."""
        return self._is_loaded and self._predictor is not None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration and paths.
        """
        self._config = config or {}
        self._checkpoint_path = self._config.get("checkpoint")
        self._label_map_path = self._config.get("label_map")
        self._is_initialized = True

    def health_check(self) -> Dict[str, Any]:
        """
        Return health status of adapter and underlying model.
        """
        status = "HEALTHY" if self.is_loaded else "UNLOADED"
        return {
            "modality": self.modality,
            "initialized": self._is_initialized,
            "loaded": self._is_loaded,
            "status": status,
            "checkpoint_path": self._checkpoint_path,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata dictionary for the adapter.
        Subclasses override to provide supported disease classes.
        """
        return {
            "modality": self.modality,
            "supported_classes": [],
            "model_version": "0.14.2.2",
            "framework": "PyTorch",
            "status": "LOADED" if self.is_loaded else "REGISTERED",
            "health": "HEALTHY" if self.is_loaded else "UNKNOWN",
            "checkpoint_path": self._checkpoint_path,
            "export_formats": ["PyTorch", "ONNX", "TorchScript"],
        }

    def shutdown(self) -> None:
        """
        Unload model weights and free memory.
        """
        self._predictor = None
        self._is_loaded = False
