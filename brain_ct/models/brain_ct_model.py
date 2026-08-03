"""
brain_ct/models/brain_ct_model.py
==================================
End-to-end PyTorch Brain CT Disease Classifier.
Reuses BackboneFactory and ModularClassifierHead from Module 6. Zero duplication.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from modality_classifier.models.backbone_factory import BackboneFactory
from modality_classifier.models.classifier_head import ModularClassifierHead
from brain_ct.utils.config import BrainCTModelConfig

logger = logging.getLogger(__name__)


class BrainCTClassifier(nn.Module):
    """
    Brain CT Disease Classification Neural Network.
    Architecture: Backbone → AdaptiveAvgPool2d → BatchNorm → Dropout → Dense Softmax.
    Number of output neurons is dynamically determined from discovered disease classes.
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
        return self.head(pooled)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature maps before pooling (for Grad-CAM readiness)."""
        return self.backbone(x)


def build_brain_ct_model(config: BrainCTModelConfig, num_classes: int) -> BrainCTClassifier:
    """Factory helper to instantiate BrainCTClassifier from config."""
    return BrainCTClassifier(
        num_classes=num_classes,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm,
    )
