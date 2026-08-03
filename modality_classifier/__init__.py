"""
modality_classifier
===================
Medical Image Modality Classification Framework (Module 6).

Intelligent routing engine identifying image modality (Brain MRI, Brain CT, Chest X-ray,
Retina, Skin, Ultrasound, Histopathology) and routing to downstream disease-specific modules.
"""

from modality_classifier.config import ModalityClassifierConfig
from modality_classifier.datasets.dataset_builder import ModalityDatasetBuilder
from modality_classifier.models.base_model import ModalityClassifier, build_modality_classifier
from modality_classifier.training.trainer import ModalityTrainer
from modality_classifier.evaluation.evaluator import ModalityEvaluator
from modality_classifier.inference.predictor import ModalityPredictor
from modality_classifier.inference.onnx_exporter import ONNXExporter

__version__ = "0.6.0"
__all__ = [
    "ModalityClassifierConfig",
    "ModalityDatasetBuilder",
    "ModalityClassifier",
    "build_modality_classifier",
    "ModalityTrainer",
    "ModalityEvaluator",
    "ModalityPredictor",
    "ONNXExporter",
]
