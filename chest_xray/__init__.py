"""
chest_xray package initialization.
Module 8: Chest X-ray Disease Analysis Framework.
"""

from chest_xray.utils.config import ChestXrayConfig
from chest_xray.datasets.dataset_loader import ChestXrayDatasetLoader
from chest_xray.models.chest_xray_model import ChestXrayClassifier
from chest_xray.training.trainer import ChestXrayTrainer
from chest_xray.evaluation.evaluator import ChestXrayEvaluator
from chest_xray.inference.predict_chest_xray import ChestXrayPredictor

__version__ = "0.8.0"

__all__ = [
    "ChestXrayConfig",
    "ChestXrayDatasetLoader",
    "ChestXrayClassifier",
    "ChestXrayTrainer",
    "ChestXrayEvaluator",
    "ChestXrayPredictor",
]
