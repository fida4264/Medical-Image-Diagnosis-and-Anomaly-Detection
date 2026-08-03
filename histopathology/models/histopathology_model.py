"""
histopathology/models/histopathology_model.py
===============================================
End-to-end PyTorch Histopathology Cancer Classifier.
Reuses BackboneFactory and ModularClassifierHead from Module 6.
Mirrors the exact pattern of brain_mri/models/brain_mri_model.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from modality_classifier.models.backbone_factory import BackboneFactory  # Module 6 reuse
from modality_classifier.models.classifier_head import ModularClassifierHead  # Module 6 reuse

from histopathology.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class HistopathologyClassifier(nn.Module):
    """
    Histopathology Cancer Subtyping Neural Network.
    Architecture: Backbone → AdaptiveAvgPool2d → BatchNorm → Dropout → Dense Softmax.
    Mirrors BrainMRIClassifier exactly (Module 7 reference template).
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

        logger.info(
            f"HistopathologyClassifier built: backbone='{backbone_name}', "
            f"in_features={self.in_features}, num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.gap(features)
        logits = self.head(pooled)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps before pooling (for Grad-CAM)."""
        return self.backbone(x)


def build_histopathology_model(
    config: ModelConfig,
    num_classes: int,
) -> HistopathologyClassifier:
    """Factory function to build and return a HistopathologyClassifier."""
    model = HistopathologyClassifier(
        num_classes=num_classes,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )
    return model
