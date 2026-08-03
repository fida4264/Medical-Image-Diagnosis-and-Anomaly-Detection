"""
Medical AI Engine — Grad-CAM Exceptions

Module 15.3: Grad-CAM Engine

Typed exception hierarchy for Grad-CAM hook registration, activation extraction,
gradient computation, tensor shape validation, and model compatibility failures.
"""

from __future__ import annotations

from medical_ai_engine.explainability.registry.cam_exceptions import CAMException


class GradCAMException(CAMException):
    """Base exception for all Grad-CAM engine errors."""

    def __init__(self, message: str, plugin_key: str = "gradcam") -> None:
        super().__init__(message=message, plugin_key=plugin_key)


class MissingTargetLayerError(GradCAMException):
    """Raised when the specified target layer cannot be found in the model."""

    def __init__(self, target_layer_name: str, available_layers: list[str] | None = None) -> None:
        avail_str = ", ".join(available_layers[:5]) if available_layers else "none"
        super().__init__(
            f"Target layer '{target_layer_name}' not found in model architecture. "
            f"Candidate layers: [{avail_str}]."
        )
        self.target_layer_name = target_layer_name
        self.available_layers = available_layers or []


class UnsupportedModelError(GradCAMException):
    """Raised when the model object does not support PyTorch autograd or forward passes."""

    def __init__(self, model_type: str, reason: str = "") -> None:
        msg = f"Model of type '{model_type}' is unsupported by Grad-CAM engine."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
        self.model_type = model_type


class HookRegistrationError(GradCAMException):
    """Raised when forward or backward hook registration on target layer fails."""

    def __init__(self, layer_name: str, reason: str) -> None:
        super().__init__(f"Failed to register PyTorch hooks on layer '{layer_name}': {reason}")
        self.layer_name = layer_name
        self.reason = reason


class GradientExtractionError(GradCAMException):
    """Raised when gradients are missing, zero, or NaN after backward pass."""

    def __init__(self, reason: str = "Gradients were not captured or contain NaN/Inf values.") -> None:
        super().__init__(f"Gradient extraction failure: {reason}")


class InvalidTensorShapeError(GradCAMException):
    """Raised when activation or gradient tensor dimensions are invalid for Grad-CAM."""

    def __init__(self, tensor_type: str, expected_dims: str, actual_shape: tuple) -> None:
        super().__init__(
            f"Invalid {tensor_type} shape {actual_shape}. Expected {expected_dims} dimensions."
        )
        self.tensor_type = tensor_type
        self.expected_dims = expected_dims
        self.actual_shape = actual_shape


class CAMComputationError(GradCAMException):
    """Raised during mathematical aggregation or channel weighting of feature maps."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Grad-CAM computation failed: {reason}")
