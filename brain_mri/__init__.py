"""
brain_mri
=========
Brain MRI Disease Analysis Framework (Module 7).

Reference implementation template for all future disease-specific AI modules (8–13).
Receives routing decision from Module 6 Modality Classifier and classifies Brain MRI diseases:
Glioma, Meningioma, Pituitary Tumor, and No Tumor.
"""

from brain_mri.utils.config import BrainMRIConfig
from brain_mri.datasets.dataset_loader import BrainMRIDatasetLoader
from brain_mri.datasets.dataset_builder import BrainMRIDataset, create_brain_mri_data_loaders
from brain_mri.models.brain_mri_model import BrainMRIClassifier, build_brain_mri_model
from brain_mri.training.trainer import BrainMRITrainer
from brain_mri.evaluation.evaluator import BrainMRIEvaluator
from brain_mri.inference.predict_brain_mri import BrainMRIPredictor
from brain_mri.reports.clinical_report import ClinicalReportGenerator
from brain_mri.export.onnx_export import BrainMRIONNXExporter
from brain_mri.export.torchscript_export import TorchScriptExporter

__version__ = "0.7.0"
__all__ = [
    "BrainMRIConfig",
    "BrainMRIDatasetLoader",
    "BrainMRIDataset",
    "create_brain_mri_data_loaders",
    "BrainMRIClassifier",
    "build_brain_mri_model",
    "BrainMRITrainer",
    "BrainMRIEvaluator",
    "BrainMRIPredictor",
    "ClinicalReportGenerator",
    "BrainMRIONNXExporter",
    "TorchScriptExporter",
]
