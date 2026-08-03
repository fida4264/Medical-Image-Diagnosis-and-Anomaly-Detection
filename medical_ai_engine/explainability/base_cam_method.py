"""
Medical AI Engine — Base CAM Method

Module 15.1: Explainability Framework Architecture

Abstract base class for all Class Activation Mapping (CAM) and saliency visualization plugins.
Includes stubs for future implementations (Grad-CAM, Grad-CAM++, LayerCAM, ScoreCAM,
EigenCAM, Integrated Gradients, Guided Backprop).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from medical_ai_engine.explainability.interfaces.cam_method_interface import ICAMMethod

logger = logging.getLogger("BaseCAMMethod")


class BaseCAMMethod(ICAMMethod, ABC):
    """
    Abstract Base Class for all CAM and Saliency method plugins in the Medical AI Platform.

    Concrete implementations (e.g. GradCAMMethod) inherit from this class and
    override `generate_activation_map`.
    """

    def __init__(self, name: str, requires_gradients: bool = True) -> None:
        self._name = name
        self._requires_gradients = requires_gradients

    @property
    def method_name(self) -> str:
        return self._name

    @property
    def requires_gradients(self) -> bool:
        return self._requires_gradients

    def generate_activation_map(
        self,
        model: Any,
        target_layer: Any,
        input_tensor: Any,
        target_category: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Base implementation stub. Concrete subclasses implement the algorithm in Module 15.2.
        """
        logger.warning(
            "BaseCAMMethod.generate_activation_map invoked on stub method '%s'. "
            "Returning uniform dummy activation map for architectural validation.",
            self._name,
        )
        # Architectural stub: returns a normalized 224x224 dummy activation map
        h, w = 224, 224
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        d = np.sqrt(x * x + y * y)
        sigma = 0.5
        g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        return (g - g.min()) / (g.max() - g.min() + 1e-8)

    def configure(self, config: Any) -> None:
        """Default no-op config injection."""
        pass

    def on_initialize(self) -> None:
        pass

    def on_ready(self) -> None:
        pass

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_unload(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "method_name": self._name,
            "requires_gradients": self._requires_gradients,
            "is_stub": True,
            "description": f"Base contract stub for {self._name}",
        }


# ── Future Method Stubs (No algorithm logic — architecture declarations only) ──

class GradCAMMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="gradcam", requires_gradients=True)


class GradCAMPlusPlusMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="gradcam_plus_plus", requires_gradients=True)


class LayerCAMMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="layercam", requires_gradients=True)


class ScoreCAMMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="scorecam", requires_gradients=False)


class EigenCAMMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="eigencam", requires_gradients=False)


class IntegratedGradientsMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="integrated_gradients", requires_gradients=True)


class GuidedBackpropMethodStub(BaseCAMMethod):
    def __init__(self) -> None:
        super().__init__(name="guided_backprop", requires_gradients=True)
