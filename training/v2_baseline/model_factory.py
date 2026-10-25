"""
training/v2_baseline/model_factory.py
=====================================
Model builder for V2.0 baseline models using transfer learning backbones (ResNet-18, EfficientNet-B0).
Wraps existing BackboneFactory or torchvision models to construct reproducible classification networks.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class V2BaselineModel(nn.Module):
    """
    Standard transfer-learning baseline classification model.
    Outputs raw logits of shape (B, num_classes).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        model_name: str = "resnet18",
        task: str = "multiclass",
    ):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.num_classes = num_classes
        self.task = task

        # Adaptive pooling ensures consistent feature vector size regardless of spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


def create_baseline_model(
    model_name: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
    task: str = "multiclass",
) -> V2BaselineModel:
    """
    Instantiates transfer learning baseline model (ResNet-18 or EfficientNet-B0).

    Args:
        model_name: 'resnet18' or 'efficientnet_b0'
        num_classes: Number of output logits (e.g., 8 for multi-label Retina)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        dropout: Dropout rate before final classification layer
        task: 'multiclass', 'binary', or 'multilabel'

    Returns:
        V2BaselineModel instance.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        feature_dim = base.fc.in_features
        # Extract convolutional backbone up to before average pool and fc
        backbone = nn.Sequential(*list(base.children())[:-2])

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        feature_dim = base.classifier[1].in_features
        backbone = base.features

    else:
        raise ValueError(
            f"Unsupported baseline model '{model_name}'. Supported: ['resnet18', 'efficientnet_b0']"
        )

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info(f"Froze backbone parameters for {model_name}.")

    model = V2BaselineModel(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        model_name=model_name,
        task=task,
    )
    logger.info(
        f"Created V2BaselineModel: {model_name} -> {num_classes} classes "
        f"(pretrained={pretrained}, task={task})"
    )
    return model
