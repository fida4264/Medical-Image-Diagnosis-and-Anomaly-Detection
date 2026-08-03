"""
skin/__init__.py
=================
Package init for Module 12: Skin Lesion Cancer Classification Framework.
Exposes the primary public API for external use.
"""

from skin.datasets.dataset_loader import SkinLesionDatasetLoader
from skin.datasets.dataset_builder import SkinLesionDataset, create_skin_data_loaders
from skin.models.skin_model import SkinLesionClassifier, build_skin_model
from skin.training.trainer import SkinLesionTrainer
from skin.evaluation.evaluator import SkinLesionEvaluator
from skin.inference.predict_skin import SkinLesionPredictor, SkinPredictionResult, PredictionReliabilityScorer
from skin.reports.clinical_report import ClinicalReportGenerator
from skin.utils.config import SkinLesionConfig

__all__ = [
    "SkinLesionDatasetLoader",
    "SkinLesionDataset",
    "create_skin_data_loaders",
    "SkinLesionClassifier",
    "build_skin_model",
    "SkinLesionTrainer",
    "SkinLesionEvaluator",
    "SkinLesionPredictor",
    "SkinPredictionResult",
    "PredictionReliabilityScorer",
    "ClinicalReportGenerator",
    "SkinLesionConfig",
]

__version__ = "0.12.0"
__module__ = "Module 12: Skin Lesion Cancer Classification Framework"
