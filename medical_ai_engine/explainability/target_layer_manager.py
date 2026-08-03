"""
Medical AI Engine — Target Layer Manager

Module 15.1: Explainability Framework Architecture

Resolves target feature extraction layers for CNN and ViT architectures across all
supported medical imaging modalities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration

logger = logging.getLogger("TargetLayerManager")


class TargetLayerManager:
    """
    Manages target layer selection rules and dynamic introspection of model architectures.
    """

    def __init__(self, config: Optional[ExplainabilityConfiguration] = None) -> None:
        self.config = config or ExplainabilityConfiguration.default()

    def resolve_target_layer(
        self,
        model: Any,
        modality: str,
        explicit_layer_name: Optional[str] = None,
    ) -> Any:
        """
        Resolve the target PyTorch / ONNX layer module object for activation map extraction.

        Args:
            model: PyTorch module instance or predictor reference.
            modality: Medical imaging modality string (e.g. 'brain_mri', 'chest_xray').
            explicit_layer_name: Optional user override for layer name.

        Returns:
            Resolved layer object/reference, or None if architectural stub.
        """
        layer_name = explicit_layer_name or self.config.get_target_layer_for_modality(modality)

        if model is None:
            logger.warning("[TargetLayerManager] Model is None. Returning layer name stub: '%s'", layer_name)
            return layer_name

        # If model has named_modules (PyTorch nn.Module)
        if hasattr(model, "named_modules"):
            for name, module in model.named_modules():
                if name == layer_name:
                    logger.debug("[TargetLayerManager] Found layer '%s' in model.", name)
                    return module
            # Fallback: find last Conv2d module
            last_conv = None
            for name, module in model.named_modules():
                if module.__class__.__name__ in ("Conv2d", "Conv3d", "LayerNorm"):
                    last_conv = module
            if last_conv is not None:
                logger.info("[TargetLayerManager] Target layer '%s' not found. Using fallback: %s", layer_name, last_conv)
                return last_conv

        return layer_name

    def list_candidate_layers(self, model: Any) -> List[str]:
        """List named candidate feature extraction layers for a model."""
        if hasattr(model, "named_modules"):
            candidates = []
            for name, module in model.named_modules():
                if module.__class__.__name__ in ("Conv2d", "Conv3d", "Sequential", "LayerNorm", "ResidualBlock"):
                    if name:
                        candidates.append(f"{name} ({module.__class__.__name__})")
            return candidates
        return ["layer4 (default)", "features (default)"]
