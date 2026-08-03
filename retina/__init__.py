"""
retina/__init__.py
====================
Package init for Module 11: Retina Disease Analysis Framework.
Exposes the primary public API for external use.
"""

from retina.datasets.dataset_loader import RetinaDatasetLoader
from retina.datasets.dataset_builder import RetinaDataset, create_retina_data_loaders
from retina.models.retina_model import RetinaClassifier, build_retina_model
from retina.training.trainer import RetinaTrainer
from retina.evaluation.evaluator import RetinaEvaluator
from retina.inference.predict_retina import RetinaPredictor, RetinaPredictionResult, SeverityGrader
from retina.reports.clinical_report import ClinicalReportGenerator
from retina.utils.config import RetinaConfig

__all__ = [
    "RetinaDatasetLoader",
    "RetinaDataset",
    "create_retina_data_loaders",
    "RetinaClassifier",
    "build_retina_model",
    "RetinaTrainer",
    "RetinaEvaluator",
    "RetinaPredictor",
    "RetinaPredictionResult",
    "SeverityGrader",
    "ClinicalReportGenerator",
    "RetinaConfig",
]

__version__ = "0.11.0"
__module__ = "Module 11: Retina Disease Analysis Framework"
