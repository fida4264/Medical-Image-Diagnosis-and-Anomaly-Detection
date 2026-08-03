"""
brain_mri/models/brain_mri_model.py
===================================
End-to-end PyTorch Brain MRI Disease Classifier module.
Reuses BackboneFactory and ModularClassifierHead from Module 6. Zero duplication.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from modality_classifier.models.backbone_factory import BackboneFactory
from modality_classifier.models.classifier_head import ModularClassifierHead
from brain_mri.utils.config import BrainMRIModelConfig

logger = logging.getLogger(__name__)


class BrainMRIClassifier(nn.Module):
    """
    Brain MRI Disease Classification Neural Network.
    Architecture: Backbone -> AdaptiveAvgPool2d -> BatchNorm -> Dropout -> Dense Softmax.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
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
        Extracts spatial feature maps before pooling for Grad-CAM.
        """
        return self.backbone(x)


def build_brain_mri_model(config: BrainMRIModelConfig, num_classes: int) -> BrainMRIClassifier:
    """
    Factory helper to instantiate BrainMRIClassifier from config.
    """
    model = BrainMRIClassifier(
        num_classes=num_classes,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
    )
    return model
