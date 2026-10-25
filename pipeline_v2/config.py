"""
pipeline_v2/config.py
=====================
Configuration and constants for V2.0 Phase 9 Pipeline Integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

# Paths to models & checkpoints (PRESERVED - NEVER OVERWRITTEN)
BASELINE_MODELS_DIR = ROOT / "models" / "v2_baseline"
CALIBRATED_MODELS_DIR = ROOT / "models" / "v2_calibrated"
OOD_MODELS_DIR = ROOT / "models" / "v2_ood"

# Paths to calibration & threshold artifacts
CALIBRATION_PARAMS_PATH = ROOT / "outputs" / "v2_phase7" / "calibration_parameters.json"
RETINA_THRESHOLDS_PATH = ROOT / "outputs" / "v2_phase7" / "retina_thresholds.json"
OOD_THRESHOLDS_PATH = ROOT / "outputs" / "v2_phase8" / "thresholds" / "ood_thresholds.json"
DATA_READINESS_PATH = ROOT / "outputs" / "v2_phase8" / "readiness" / "phase8_data_readiness.json"

# Phase 9 Outputs
OUTPUTS_P9_DIR = ROOT / "outputs" / "v2_phase9"
AUDIT_DIR = OUTPUTS_P9_DIR / "audit"
REPORTS_DIR = OUTPUTS_P9_DIR / "reports"
AUDIT_LOG_FILE = AUDIT_DIR / "inference_audit.jsonl"

for d in [OUTPUTS_P9_DIR, AUDIT_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PIPELINE_VERSION = "V2.0-Phase9"
SEED = 42

# Canonical 7 Modalities
SUPPORTED_MODALITIES: List[str] = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "histopathology",
    "retina",
    "skin",
    "ultrasound",
]

# Modality metadata detailing classes, tasks, and configurations
MODALITY_SETTINGS: Dict[str, Dict[str, Any]] = {
    "brain_mri": {
        "display_name": "Brain MRI",
        "task": "multiclass",
        "num_classes": 4,
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "normal_class": "no_tumor",
        "normal_idx": 2,
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "brain_mri" / "best_model.pt",
    },
    "brain_ct": {
        "display_name": "Brain CT",
        "task": "binary",
        "num_classes": 2,
        "classes": ["normal", "hemorrhage"],
        "normal_class": "normal",
        "normal_idx": 0,
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "brain_ct" / "fold_0" / "best_model.pt",
        "cv_limitation": "5-fold CV Fold 0 model; multi-center validation pending",
    },
    "chest_xray": {
        "display_name": "Chest X-ray",
        "task": "multiclass",
        "num_classes": 4,
        "classes": ["covid", "lung_opacity", "normal", "viral_pneumonia"],
        "normal_class": "normal",
        "normal_idx": 2,
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "chest_xray" / "best_model.pt",
    },
    "histopathology": {
        "display_name": "Histopathology",
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
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "histopathology" / "best_model.pt",
    },
    "retina": {
        "display_name": "Retinal Fundus",
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
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "retina" / "best_model.pt",
        "multi_label": True,
    },
    "skin": {
        "display_name": "Skin Lesion (Dermoscopy)",
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
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "skin" / "best_model.pt",
    },
    "ultrasound": {
        "display_name": "Breast Ultrasound",
        "task": "multiclass",
        "num_classes": 3,
        "classes": ["benign", "malignant", "normal"],
        "normal_class": "normal",
        "normal_idx": 2,
        "input_size": (224, 224),
        "channels": 3,
        "checkpoint": BASELINE_MODELS_DIR / "ultrasound" / "fold_0" / "best_model.pt",
        "cv_limitation": "5-fold CV Fold 0 model; multi-center validation pending",
    },
}

# Image validation constraints
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm"}
MIN_IMAGE_DIM = 16
MAX_IMAGE_DIM = 10000
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Mandatory Regulatory / Research Disclaimer
LEGAL_LIMITATIONS_NOTICE = (
    "RESEARCH PROTOTYPE ONLY — NOT FOR DIRECT CLINICAL USE. "
    "This system is an investigational deep learning research platform. "
    "Predictions, calibrated confidence scores, OOD detections, and anomaly scores "
    "are non-diagnostic computational estimates. Abstentions or quality gate rejections "
    "do NOT establish presence or absence of pathology."
)
