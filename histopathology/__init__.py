"""
histopathology/__init__.py
============================
Package init for Module 10: Histopathology Cancer Subtyping Framework.
Exposes the primary public API for external use.
"""

from histopathology.datasets.dataset_loader import HistopathologyDatasetLoader
from histopathology.datasets.dataset_builder import HistopathologyDataset, create_histopathology_data_loaders
from histopathology.models.histopathology_model import HistopathologyClassifier, build_histopathology_model
from histopathology.training.trainer import HistopathologyTrainer
from histopathology.evaluation.evaluator import HistopathologyEvaluator
from histopathology.inference.predict_histopathology import HistopathologyPredictor, HistopathologyPredictionResult
from histopathology.reports.clinical_report import ClinicalReportGenerator
from histopathology.utils.config import HistopathologyConfig

__all__ = [
    "HistopathologyDatasetLoader",
    "HistopathologyDataset",
    "create_histopathology_data_loaders",
    "HistopathologyClassifier",
    "build_histopathology_model",
    "HistopathologyTrainer",
    "HistopathologyEvaluator",
    "HistopathologyPredictor",
    "HistopathologyPredictionResult",
    "ClinicalReportGenerator",
    "HistopathologyConfig",
]

__version__ = "0.10.0"
__module__ = "Module 10: Histopathology Cancer Subtyping Framework"
