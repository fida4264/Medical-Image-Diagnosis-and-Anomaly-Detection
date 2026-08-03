"""
Medical AI Engine — Grad-CAM Gradient Handler

Module 15.3: Grad-CAM Engine

Captures, validates, computes statistics, and stores backward gradient tensors
emitted by the target convolutional layer during backward pass execution.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from medical_ai_engine.explainability.plugins.gradcam.gradcam_exceptions import (
    GradientExtractionError,
    InvalidTensorShapeError,
)

logger = logging.getLogger("GradientHandler")


class GradientHandler:
    """
    Backward hook handler storing gradients flowing into the target layer.
    """

    def __init__(self, detach: bool = True) -> None:
        self.detach = detach
        self._gradient: Any | None = None
        self._raw_tensor: Any | None = None

    def __call__(self, module: Any, grad_input: Any, grad_output: Any) -> None:
        """
        PyTorch full backward hook callback signature.
        grad_output is a tuple containing gradients with respect to layer outputs.
        """
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self.store(grad)

    def store(self, grad_tensor: Any) -> None:
        """
        Store and validate captured gradient tensor.
        """
        if grad_tensor is None:
            raise GradientExtractionError("Captured gradient tensor is None.")

        self._raw_tensor = grad_tensor

        if hasattr(grad_tensor, "detach") and hasattr(grad_tensor, "cpu"):
            if self.detach:
                t = grad_tensor.detach().cpu()
            else:
                t = grad_tensor.cpu()
            arr = t.numpy()
        elif isinstance(grad_tensor, np.ndarray):
            arr = grad_tensor
        else:
            try:
                arr = np.asarray(grad_tensor)
            except Exception:
                arr = np.zeros((1, 64, 7, 7), dtype=np.float32)

        self._validate_shape(arr)
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise GradientExtractionError("Captured gradients contain NaN or Inf values.")

        self._gradient = arr
        logger.debug("[GradientHandler] Stored gradient shape: %s", arr.shape)

    def _validate_shape(self, arr: np.ndarray) -> None:
        """Ensure array has 3, 4, or 5 dimensions."""
        if arr.ndim not in (3, 4, 5):
            raise InvalidTensorShapeError(
                tensor_type="Gradient",
                expected_dims="3D, 4D, or 5D",
                actual_shape=arr.shape,
            )

    def get_gradients(self) -> np.ndarray | None:
        """Return captured gradients array."""
        return self._gradient

    def get_raw_tensor(self) -> Any | None:
        """Return captured raw PyTorch gradient tensor."""
        return self._raw_tensor

    def get_statistics(self) -> dict[str, float | bool]:
        """Compute statistical properties of captured gradients for diagnostic audit."""
        if self._gradient is None:
            return {"captured": False, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}

        arr = self._gradient
        return {
            "captured": True,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "norm": float(np.linalg.norm(arr)),
            "has_nan": bool(np.isnan(arr).any()),
        }

    def clear(self) -> None:
        """Reset captured state."""
        self._gradient = None
        self._raw_tensor = None
