"""
modality_classifier/models
==========================
Model definitions, transfer learning backbones, and classification heads.
"""

from modality_classifier.models.base_model import ModalityClassifier, build_modality_classifier
from modality_classifier.models.backbone_factory import BackboneFactory
from modality_classifier.models.classifier_head import ModularClassifierHead

__all__ = [
    "ModalityClassifier",
    "build_modality_classifier",
    "BackboneFactory",
    "ModularClassifierHead",
]
