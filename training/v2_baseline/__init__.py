"""
training/v2_baseline/__init__.py
================================
V2.0 Shared Baseline Model Training & Experiment Framework.
Provides modular, reproducible training and evaluation for all 7 medical imaging modalities.
"""

from .config import ExperimentConfig, ModalityConfig
from .dataset import V2ManifestDataset
from .transforms import get_transforms
from .model_factory import create_baseline_model
from .loss_factory import create_loss_function, compute_class_weights, compute_multilabel_pos_weights
from .metrics import calculate_metrics, calculate_multilabel_metrics
from .trainer import V2Trainer
from .experiment import ExperimentRegistry

__all__ = [
    "ExperimentConfig",
    "ModalityConfig",
    "V2ManifestDataset",
    "get_transforms",
    "create_baseline_model",
    "create_loss_function",
    "compute_class_weights",
    "compute_multilabel_pos_weights",
    "calculate_metrics",
    "calculate_multilabel_metrics",
    "V2Trainer",
    "ExperimentRegistry",
]
