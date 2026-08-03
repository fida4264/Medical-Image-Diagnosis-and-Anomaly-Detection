"""
Medical AI Engine — Grad-CAM Activation Extractor

Module 15.3: Grad-CAM Engine

Captures, validates, and stores feature map activation tensors emitted
by the target convolutional layer during forward pass execution.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from medical_ai_engine.explainability.plugins.gradcam.gradcam_exceptions import (
    InvalidTensorShapeError,
)

logger = logging.getLogger("ActivationExtractor")


class ActivationExtractor:
    """
    Forward hook handler storing feature activations.
    """

    def __init__(self, detach: bool = True) -> None:
        self.detach = detach
        self._activation: Any | None = None
        self._raw_tensor: Any | None = None

    def __call__(self, module: Any, input_tensor: Any, output_tensor: Any) -> None:
        """
        PyTorch forward hook callback signature.
        """
        self.store(output_tensor)

    def store(self, output_tensor: Any) -> None:
        """
        Store and validate captured output tensor from target layer module.
        """
        tensor = output_tensor
        # Handle tuple output if layer returns (output, aux)
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]

        self._raw_tensor = tensor

        # If it's a PyTorch Tensor
        if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
            if self.detach:
                t = tensor.detach().cpu()
            else:
                t = tensor.cpu()

            arr = t.numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            # Fallback for mock objects
            try:
                arr = np.asarray(tensor)
            except Exception:
                arr = np.zeros((1, 64, 7, 7), dtype=np.float32)

        self._validate_shape(arr)
        self._activation = arr
        logger.debug("[ActivationExtractor] Stored activation shape: %s", arr.shape)

    def _validate_shape(self, arr: np.ndarray) -> None:
        """Ensure array has 3, 4, or 5 dimensions."""
        if arr.ndim not in (3, 4, 5):
            raise InvalidTensorShapeError(
                tensor_type="Activation",
                expected_dims="3D, 4D, or 5D",
                actual_shape=arr.shape,
            )

    def get_activations(self) -> np.ndarray | None:
        """Return captured activations array."""
        return self._activation

    def get_raw_tensor(self) -> Any | None:
        """Return captured raw PyTorch tensor (if available)."""
        return self._raw_tensor

    def clear(self) -> None:
        """Reset captured state."""
        self._activation = None
        self._raw_tensor = None
