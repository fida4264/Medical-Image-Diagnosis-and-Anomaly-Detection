"""
modality_classifier/models/base_model.py
=========================================
PyTorch Modality Classifier neural network module combining backbone and classification head.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from modality_classifier.config import ModelConfig
from modality_classifier.models.backbone_factory import BackboneFactory
from modality_classifier.models.classifier_head import ModularClassifierHead

logger = logging.getLogger(__name__)


class ModalityClassifier(nn.Module):
    """
    End-to-end PyTorch Modality Classifier neural network.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.backbone, self.in_features = BackboneFactory.create_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = ModularClassifierHead(
            in_features=self.in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.gap(features)
        logits = self.head(pooled)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial feature maps before pooling (for Grad-CAM & Explainable AI).
        """
        return self.backbone(x)


def build_modality_classifier(
    config: ModelConfig,
    num_classes: int,
) -> ModalityClassifier:
    """
    Factory helper to instantiate ModalityClassifier from ModelConfig.
    """
    model = ModalityClassifier(
        num_classes=num_classes,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
    )
    return model
