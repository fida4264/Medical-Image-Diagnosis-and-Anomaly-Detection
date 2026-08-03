"""
Medical AI Engine — CAM Method Interface

Module 15.1: Explainability Framework Architecture

Defines the contract that all Class Activation Mapping (CAM) and saliency
visualization methods must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ICAMMethod(ABC):
    """
    Abstract interface for Class Activation Mapping (CAM) visualization methods.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique identifier of the CAM method (e.g. 'gradcam', 'gradcam_plus_plus')."""
        pass

    @property
    @abstractmethod
    def requires_gradients(self) -> bool:
        """Returns True if the method requires backward gradient computation."""
        pass

    @abstractmethod
    def generate_activation_map(
        self,
        model: Any,
        target_layer: Any,
        input_tensor: Any,
        target_category: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Generate a normalized 2D activation map (values in [0, 1]).

        Args:
            model: Neural network model instance (PyTorch / ONNX).
            target_layer: Feature extraction layer reference.
            input_tensor: Preprocessed input tensor (batch, channels, H, W).
            target_category: Target class index to explain (defaults to top-1).
            extra_params: Optional method-specific parameters.

        Returns:
            2D numpy array of shape (H, W) with float values in [0.0, 1.0].
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata regarding method capability and requirements."""
        pass
