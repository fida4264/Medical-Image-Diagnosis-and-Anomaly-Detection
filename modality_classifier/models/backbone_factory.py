"""
modality_classifier/models/backbone_factory.py
===============================================
Factory for creating transfer learning feature extraction backbones using PyTorch torchvision.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class BackboneFactory:
    """
    Factory creating pretrained feature extraction backbones.
    """

    SUPPORTED_BACKBONES = {
        "mobilenet_v3_small",
        "resnet18",
        "resnet34",
        "efficientnet_b0",
        # Added in Module 7 — available to all future disease-specific modules
        "densenet121",
        "convnext_tiny",
    }

    @staticmethod
    def create_backbone(
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> Tuple[nn.Module, int]:
        """
        Instantiates feature extractor backbone and returns (backbone_module, out_features).
        Supports: mobilenet_v3_small, resnet18, resnet34, efficientnet_b0, densenet121, convnext_tiny.
        """
        name_lower = name.lower()
        if name_lower not in BackboneFactory.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{name}'. "
                f"Supported: {sorted(list(BackboneFactory.SUPPORTED_BACKBONES))}"
            )

        if name_lower == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            out_features = model.classifier[0].in_features
            backbone = model.features

        elif name_lower == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            out_features = model.fc.in_features
            backbone = nn.Sequential(*list(model.children())[:-2])

        elif name_lower == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            out_features = model.fc.in_features
            backbone = nn.Sequential(*list(model.children())[:-2])

        elif name_lower == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            out_features = model.classifier[1].in_features
            backbone = model.features

        elif name_lower == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            out_features = model.classifier.in_features
            backbone = model.features

        elif name_lower == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            out_features = model.classifier[2].in_features
            backbone = model.features

        else:
            raise NotImplementedError(f"Backbone {name_lower} implementation missing.")

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            logger.info(f"Froze feature extractor backbone parameters for {name_lower}.")

        logger.info(f"Built backbone '{name_lower}' (pretrained={pretrained}) with out_features={out_features}.")
        return backbone, out_features
