"""
Medical AI Engine — Grad-CAM Model Hook Manager

Module 15.3: Grad-CAM Engine

Safely attaches and removes PyTorch forward and backward hooks on target layer modules.
Handles version differences (register_full_backward_hook vs register_backward_hook)
and guarantees hook cleanup via context manager or explicit remove_hooks().
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from medical_ai_engine.explainability.plugins.gradcam.gradcam_exceptions import (
    HookRegistrationError,
    MissingTargetLayerError,
)

logger = logging.getLogger("ModelHookManager")


class ModelHookManager:
    """
    RAII Context Manager for PyTorch forward and backward hooks.
    """

    def __init__(self) -> None:
        self._forward_handle: Any | None = None
        self._backward_handle: Any | None = None
        self._registered: bool = False

    def register_hooks(
        self,
        model: Any,
        target_layer_module: Any,
        forward_callback: Callable,
        backward_callback: Callable,
    ) -> None:
        """
        Attach forward and backward hooks to target_layer_module.

        Args:
            model: PyTorch nn.Module or predictor.
            target_layer_module: Resolved nn.Module layer instance.
            forward_callback: ActivationExtractor.__call__
            backward_callback: GradientHandler.__call__

        Raises:
            MissingTargetLayerError: If target_layer_module is None or invalid.
            HookRegistrationError: If PyTorch hook attachment fails.
        """
        if target_layer_module is None:
            raise MissingTargetLayerError(target_layer_name="None")

        if not hasattr(target_layer_module, "register_forward_hook"):
            logger.warning(
                "[ModelHookManager] Target layer '%s' has no register_forward_hook method. "
                "Skipping PyTorch hook registration (stub mode).",
                getattr(target_layer_module, "__class__", type(target_layer_module)).__name__,
            )
            return

        try:
            # 1. Forward hook
            self._forward_handle = target_layer_module.register_forward_hook(forward_callback)

            # 2. Backward hook (prefer register_full_backward_hook for PyTorch 1.8+)
            if hasattr(target_layer_module, "register_full_backward_hook"):
                self._backward_handle = target_layer_module.register_full_backward_hook(backward_callback)
            elif hasattr(target_layer_module, "register_backward_hook"):
                self._backward_handle = target_layer_module.register_backward_hook(backward_callback)
            else:
                logger.warning("[ModelHookManager] Backward hook not supported on module.")

            self._registered = True
            logger.debug("[ModelHookManager] Hooks successfully attached to %s", target_layer_module)
        except Exception as exc:
            self.remove_hooks()
            raise HookRegistrationError(
                layer_name=str(target_layer_module),
                reason=str(exc),
            ) from exc

    def remove_hooks(self) -> None:
        """Safely detach and remove all active PyTorch hook handles."""
        if self._forward_handle is not None:
            try:
                self._forward_handle.remove()
            except Exception as e:
                logger.debug("[ModelHookManager] Error removing forward hook: %s", e)
            self._forward_handle = None

        if self._backward_handle is not None:
            try:
                self._backward_handle.remove()
            except Exception as e:
                logger.debug("[ModelHookManager] Error removing backward hook: %s", e)
            self._backward_handle = None

        self._registered = False
        logger.debug("[ModelHookManager] Hooks removed.")

    @property
    def is_registered(self) -> bool:
        return self._registered

    def __enter__(self) -> ModelHookManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()
