"""
Medical AI Engine — Grad-CAM Plugin Implementation

Module 15.3: Grad-CAM Engine

Concrete `GradCAMPlugin` inheriting from `BaseCAMPlugin` and satisfying `ICAMPlugin`.
Integrates with the `CAMProductionRegistry` system so the ExplainabilityEngine can
dynamically request and instantiate Grad-CAM without direct imports.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np

from medical_ai_engine.explainability.plugins.base_cam_plugin import BaseCAMPlugin
from medical_ai_engine.explainability.registry.cam_configuration import CAMPluginConfiguration
from medical_ai_engine.explainability.registry.cam_metadata import (
    CAMPluginMetadata,
    PluginLifecycleState,
    PluginStatus,
)
from medical_ai_engine.explainability.plugins.gradcam.gradcam_configuration import GradCAMConfiguration
from medical_ai_engine.explainability.plugins.gradcam.gradcam_engine import GradGAMEngine

logger = logging.getLogger("GradCAMPlugin")


class GradCAMPlugin(BaseCAMPlugin):
    """
    Production Grad-CAM plugin implementation for the Medical AI Platform.

    Class Activation Mapping via Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    """

    def __init__(self) -> None:
        super().__init__(
            plugin_key="gradcam",
            plugin_version="0.15.3.0",
            requires_gradients_flag=True,
            description="Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017). "
                        "Computes feature map weights via global average pooling of backward gradients.",
        )
        self._gradcam_config = GradCAMConfiguration.default()
        self._engine: Optional[GradGAMEngine] = None

    # ── ICAMPlugin: Lifecycle Callbacks ───────────────────────────────────────

    def on_initialize(self) -> None:
        """REGISTERED → INITIALIZED. Instantiate internal GradGAMEngine."""
        super().on_initialize()
        self._engine = GradGAMEngine(config=self._gradcam_config)
        logger.info("[GradCAMPlugin] Initialized GradGAMEngine instance.")

    def on_ready(self) -> None:
        """INITIALIZED → READY."""
        super().on_ready()
        logger.debug("[GradCAMPlugin] Plugin is READY.")

    def on_activate(self) -> None:
        """READY → ACTIVE."""
        super().on_activate()
        logger.debug("[GradCAMPlugin] Plugin is ACTIVE.")

    def on_deactivate(self) -> None:
        """ACTIVE → READY."""
        super().on_deactivate()
        logger.debug("[GradCAMPlugin] Plugin deactivated.")

    def on_unload(self) -> None:
        """→ UNLOADED. Release engine reference."""
        super().on_unload()
        self._engine = None
        logger.info("[GradCAMPlugin] Unloaded.")

    # ── ICAMPlugin: Configuration ─────────────────────────────────────────────

    def configure(self, config: CAMPluginConfiguration) -> None:
        """Inject configuration from factory."""
        super().configure(config)
        self._gradcam_config = GradCAMConfiguration(
            target_layer_name=config.target_layer_name,
            target_category=config.target_category,
            relu_on_output=config.relu_on_output,
            normalize_output=config.normalize_output,
            output_height=config.output_height,
            output_width=config.output_width,
            device=config.device,
        )
        if self._engine is not None:
            self._engine.config = self._gradcam_config

    # ── ICAMPlugin: Core Algorithm ────────────────────────────────────────────

    def generate_activation_map(
        self,
        model: Any,
        target_layer: Any,
        input_tensor: Any,
        target_category: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Compute and return 2D normalized Grad-CAM heatmap array [0, 1].

        Args:
            model: PyTorch model or predictor.
            target_layer: Resolved target layer or layer name string.
            input_tensor: Image tensor (B, C, H, W).
            target_category: Target class index.
            extra_params: Optional dictionary containing 'context', 'modality', etc.

        Returns:
            2D numpy float32 array of shape (output_height, output_width) in [0.0, 1.0].
        """
        if self._engine is None:
            self._engine = GradGAMEngine(config=self._gradcam_config)

        extra = extra_params or {}
        context = extra.get("context", None)
        modality = extra.get("modality", "brain_mri")
        layer_name = target_layer if isinstance(target_layer, str) else self._gradcam_config.target_layer_name

        payload = self._engine.compute(
            model=model,
            input_tensor=input_tensor,
            modality=modality,
            target_layer_name=layer_name,
            target_category=target_category,
            context=context,
        )

        return payload["normalized_heatmap"]

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_plugin_metadata(self) -> CAMPluginMetadata:
        return CAMPluginMetadata(
            method_name="gradcam",
            version=self.plugin_version,
            author="Medical AI Platform Team",
            description=self._description,
            requires_gradients=True,
            heatmap_support=True,
            overlay_support=True,
            is_stub=False,  # Concrete production implementation!
            status=PluginStatus.STABLE,
            supported_frameworks=["pytorch"],
            supported_modalities=[
                "brain_mri", "chest_xray", "skin_lesion",
                "eye_retinal", "breast_mri", "ct_scan", "pathology",
            ],
            target_layer_compatibility=["Conv2d", "Sequential", "LayerNorm"],
            lifecycle_state=self._lifecycle_state,
        )
