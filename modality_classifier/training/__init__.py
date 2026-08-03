"""
modality_classifier/training
============================
Training loop, metrics calculation, and training callbacks.
"""

from modality_classifier.training.trainer import ModalityTrainer
from modality_classifier.training.metrics import compute_classification_metrics
from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint

__all__ = [
    "ModalityTrainer",
    "compute_classification_metrics",
    "EarlyStopping",
    "ModelCheckpoint",
]
