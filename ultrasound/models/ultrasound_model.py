"""
ultrasound/models/ultrasound_model.py
======================================
Classifier model architecture for Module 13: Ultrasound Disease Analysis Framework.
Reuses Module 6 BackboneFactory and ModularClassifierHead.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from modality_classifier.models.backbone_factory import BackboneFactory  # Module 6 reuse
from modality_classifier.models.classifier_head import ModularClassifierHead  # Module 6 reuse
from ultrasound.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class UltrasoundClassifier(nn.Module):
    """
    Ultrasound Disease Classifier.
    Input → Backbone → Global Average Pooling → BatchNorm → Dropout → Softmax
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Re-use Module 6 BackboneFactory
        self.backbone, in_features = BackboneFactory.create_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=False,
        )

        # Re-use Module 6 ModularClassifierHead
        self.head = ModularClassifierHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for visualization / Grad-CAM / future downstream tasks."""
        return self.backbone(x)


def build_ultrasound_model(config: ModelConfig, num_classes: int) -> UltrasoundClassifier:
    """Factory function for building UltrasoundClassifier from configuration."""
    model = UltrasoundClassifier(
        backbone_name=config.backbone,
        num_classes=num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )
    logger.info(
        f"Built UltrasoundClassifier with backbone='{config.backbone}', "
        f"num_classes={num_classes}, dropout={config.dropout_rate}"
    )
    return model
