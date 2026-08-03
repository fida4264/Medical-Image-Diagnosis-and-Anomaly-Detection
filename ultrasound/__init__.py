"""
ultrasound/__init__.py
=======================
Package init for Module 13: Ultrasound Disease Analysis Framework.
Exposes the primary public API for external use.
"""

from ultrasound.datasets.dataset_loader import UltrasoundDatasetLoader
from ultrasound.datasets.dataset_builder import UltrasoundDataset, create_ultrasound_data_loaders
from ultrasound.models.ultrasound_model import UltrasoundClassifier, build_ultrasound_model
from ultrasound.training.trainer import UltrasoundTrainer
from ultrasound.evaluation.evaluator import UltrasoundEvaluator
from ultrasound.inference.predict_ultrasound import UltrasoundPredictor, UltrasoundPredictionResult, PredictionReliabilityScorer
from ultrasound.reports.clinical_report import ClinicalReportGenerator
from ultrasound.utils.config import UltrasoundConfig

__all__ = [
    "UltrasoundDatasetLoader",
    "UltrasoundDataset",
    "create_ultrasound_data_loaders",
    "UltrasoundClassifier",
    "build_ultrasound_model",
    "UltrasoundTrainer",
    "UltrasoundEvaluator",
    "UltrasoundPredictor",
    "UltrasoundPredictionResult",
    "PredictionReliabilityScorer",
    "ClinicalReportGenerator",
    "UltrasoundConfig",
]

__version__ = "0.13.0"
__module__ = "Module 13: Ultrasound Disease Analysis Framework"
