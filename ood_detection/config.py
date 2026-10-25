"""
ood_detection/config.py
=======================
Central configuration for V2.0 Phase 8 Out-of-Distribution (OOD) and Anomaly Detection.
Defines directories, modality metadata, score definitions, orientation contracts,
evaluation scenarios, and reproducibility parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Root directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = MANIFESTS_DIR / "splits"

# Models directories
BASELINE_MODELS_DIR = ROOT / "models" / "v2_baseline"
CALIBRATED_MODELS_DIR = ROOT / "models" / "v2_calibrated"
OOD_MODELS_DIR = ROOT / "models" / "v2_ood"

# Phase 8 output directories
OUTPUTS_P8_DIR = ROOT / "outputs" / "v2_phase8"
READINESS_DIR = OUTPUTS_P8_DIR / "readiness"
SCORES_DIR = OUTPUTS_P8_DIR / "scores"
THRESHOLDS_DIR = OUTPUTS_P8_DIR / "thresholds"
METRICS_DIR = OUTPUTS_P8_DIR / "metrics"
PLOTS_DIR = OUTPUTS_P8_DIR / "plots"
REPORTS_DIR = OUTPUTS_P8_DIR / "reports"
CONF_MAT_DIR = REPORTS_DIR / "ood_confusion_matrices"

for d in [
    OOD_MODELS_DIR,
    OUTPUTS_P8_DIR,
    READINESS_DIR,
    SCORES_DIR,
    THRESHOLDS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    CONF_MAT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Global reproducibility seed
SEED = 42

# Canonical 7 modalities
ALL_MODALITIES = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "histopathology",
    "retina",
    "skin",
    "ultrasound",
]

MULTICLASS_MODALITIES = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "histopathology",
    "skin",
    "ultrasound",
]

# Supported OOD Scoring Methods
SUPPORTED_METHODS = [
    "msp",
    "entropy",
    "energy",
    "mahalanobis",
    "knn",
]

# SCORE ORIENTATION CONTRACT:
# For ALL scoring methods in this framework:
# HIGHER SCORE = MORE OOD / ANOMALOUS.
# LOWER SCORE = MORE IN-DISTRIBUTION (ID).
SCORE_ORIENTATION_HIGHER_IS_OOD = True

# Modality configurations detailing tasks, classes, channels, splits
MODALITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "brain_mri": {
        "task": "multiclass",
        "num_classes": 4,
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "normal_class": "no_tumor",
        "normal_idx": 2,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "fixed",
        "manifest": SPLITS_DIR / "brain_mri_v2.0_splits.csv",
    },
    "brain_ct": {
        "task": "binary",
        "num_classes": 2,
        "classes": ["normal", "hemorrhage"],
        "normal_class": "normal",
        "normal_idx": 0,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "kfold",
        "manifest": SPLITS_DIR / "brain_ct_v2.0_folds.csv",
    },
    "chest_xray": {
        "task": "multiclass",
        "num_classes": 4,
        "classes": ["covid", "lung_opacity", "normal", "viral_pneumonia"],
        "normal_class": "normal",
        "normal_idx": 2,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "fixed",
        "manifest": SPLITS_DIR / "chest_xray_v2.0_splits.csv",
    },
    "histopathology": {
        "task": "multiclass",
        "num_classes": 5,
        "classes": [
            "colon_adenocarcinoma",
            "colon_normal",
            "lung_adenocarcinoma",
            "lung_normal",
            "lung_squamous_cell_carcinoma",
        ],
        "normal_class": "colon_normal",
        "normal_idx": 1,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "fixed",
        "manifest": SPLITS_DIR / "histopathology_v2.0_splits.csv",
    },
    "retina": {
        "task": "multilabel",
        "num_classes": 8,
        "classes": [
            "normal",
            "diabetic_retinopathy",
            "glaucoma",
            "cataract",
            "amd",
            "hypertension",
            "pathological_myopia",
            "other_abnormality",
        ],
        "normal_class": "normal",
        "normal_idx": 0,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "fixed",
        "manifest": SPLITS_DIR / "retina_v2.0_splits.csv",
    },
    "skin": {
        "task": "multiclass",
        "num_classes": 7,
        "classes": [
            "melanocytic_nevus",
            "melanoma",
            "benign_keratosis",
            "basal_cell_carcinoma",
            "actinic_keratosis",
            "vascular_lesion",
            "dermatofibroma",
        ],
        "normal_class": "melanocytic_nevus",
        "normal_idx": 0,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "fixed",
        "manifest": SPLITS_DIR / "skin_v2.0_splits.csv",
    },
    "ultrasound": {
        "task": "multiclass",
        "num_classes": 3,
        "classes": ["benign", "malignant", "normal"],
        "normal_class": "normal",
        "normal_idx": 2,
        "channels": 3,
        "input_size": (224, 224),
        "split_type": "kfold",
        "manifest": SPLITS_DIR / "ultrasound_v2.0_folds.csv",
    },
}

SCORE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "msp": {
        "name": "Maximum Softmax Probability (Complement)",
        "formula": "1.0 - max_c p_c (or 1.0 - max_c sigma(z_c) for multi-label)",
        "orientation": "higher_is_ood",
        "reference": "Hendrycks & Gimpel (ICLR 2017)",
        "range": "[0, 1]",
    },
    "entropy": {
        "name": "Predictive Entropy",
        "formula": "- sum_c p_c * log(p_c) (or binary entropy average for multi-label)",
        "orientation": "higher_is_ood",
        "reference": "Lakshminarayanan et al. (NeurIPS 2017)",
        "range": "[0, log(C)] (normalized to [0, 1])",
    },
    "energy": {
        "name": "Free Energy Score",
        "formula": "- T * log(sum_c exp(z_c / T)) with T=1.0",
        "orientation": "higher_is_ood",
        "reference": "Liu et al. (NeurIPS 2020)",
        "range": "(-inf, inf)",
    },
    "mahalanobis": {
        "name": "Class-Conditional Mahalanobis Distance",
        "formula": "min_c sqrt((z - mu_c)^T * Sigma^-1 * (z - mu_c))",
        "orientation": "higher_is_ood",
        "reference": "Lee et al. (NeurIPS 2018)",
        "range": "[0, inf)",
    },
    "knn": {
        "name": "K-Nearest Neighbors Feature Distance",
        "formula": "mean distance to k nearest training feature representations (k=5)",
        "orientation": "higher_is_ood",
        "reference": "Sun et al. (ICML 2022)",
        "range": "[0, inf)",
    },
}

# Evaluation Scenarios
SCENARIOS = {
    "scenario_a": "Cross-Modality OOD (mismatched medical modality inputs)",
    "scenario_b": "Unrelated Image OOD (synthetic/non-medical natural patterns)",
    "scenario_c": "Medical Distribution Shift (cross-scanner / intensity shifts)",
    "scenario_d": "Supported-Class Model Uncertainty (ambiguous in-distribution inputs)",
}

# Normal Reference Classes for Anomaly Detection (within-modality)
NORMAL_CLASS_NAMES: Dict[str, str] = {
    "brain_mri": "no_tumor",
    "brain_ct": "normal",
    "chest_xray": "normal",
    "retina": "normal",
    "ultrasound": "normal",
}
