"""
metadata_manager/config.py
===========================
Centralised configuration for Module 2: Dataset Standardization & Metadata Management.

All paths, column names, modality mappings, class normalization rules, and
per-dataset extraction strategies are defined here so every other module
imports from a single source of truth.

Author: Medical AI System — Module 2
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Root & I/O Directories
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Module 1 outputs (inputs to this module)
MODULE1_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
METADATA_CSV: Path = MODULE1_OUTPUTS_DIR / "metadata.csv"
LABELS_CSV: Path = MODULE1_OUTPUTS_DIR / "labels.csv"
STATISTICS_JSON: Path = MODULE1_OUTPUTS_DIR / "dataset_statistics.json"
VALIDATION_REPORT_JSON: Path = MODULE1_OUTPUTS_DIR / "validation_report.json"
CLASS_DISTRIBUTION_CSV: Path = MODULE1_OUTPUTS_DIR / "class_distribution.csv"

# Module 2 outputs
METADATA_OUTPUT_DIR: Path = PROJECT_ROOT / "metadata"
VISUALIZATIONS_DIR: Path = METADATA_OUTPUT_DIR / "visualizations"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE_NAME: str = "metadata_manager.log"

# Output file names
MASTER_METADATA_CSV: str = "master_metadata.csv"
IMAGE_MAPPING_CSV: str = "image_mapping.csv"
CLASS_MAPPING_CSV: str = "class_mapping.csv"
MODALITY_MAPPING_CSV: str = "modality_mapping.csv"
DATASET_REGISTRY_JSON: str = "dataset_registry.json"
DATASET_SUMMARY_MD: str = "dataset_summary.md"

# ---------------------------------------------------------------------------
# Column Names — master_metadata.csv
# ---------------------------------------------------------------------------

COL_IMAGE_ID = "image_id"
COL_INTERNAL_ID = "internal_id"
COL_DATASET_NAME = "dataset_name"
COL_DATASET_SOURCE = "dataset_source"
COL_MODALITY = "modality"
COL_DISEASE = "disease"
COL_NORMALIZED_CLASS = "normalized_class"
COL_ORIGINAL_CLASS = "original_class"
COL_SPLIT = "split"
COL_IMAGE_PATH = "image_path"
COL_EXTENSION = "extension"
COL_WIDTH = "width"
COL_HEIGHT = "height"
COL_CHANNELS = "channels"
COL_FILE_SIZE = "file_size_bytes"
COL_IS_VALID = "is_valid"
COL_MD5_HASH = "md5_hash"
COL_LESION_ID = "lesion_id"

MASTER_METADATA_COLUMNS = [
    COL_IMAGE_ID,
    COL_INTERNAL_ID,
    COL_DATASET_NAME,
    COL_DATASET_SOURCE,
    COL_MODALITY,
    COL_DISEASE,
    COL_NORMALIZED_CLASS,
    COL_ORIGINAL_CLASS,
    COL_SPLIT,
    COL_IMAGE_PATH,
    COL_EXTENSION,
    COL_WIDTH,
    COL_HEIGHT,
    COL_CHANNELS,
    COL_FILE_SIZE,
    COL_IS_VALID,
    COL_MD5_HASH,
    COL_LESION_ID,
]

# ---------------------------------------------------------------------------
# Modality Normalization
# Raw modality folder names → canonical lowercase modality name
# ---------------------------------------------------------------------------

MODALITY_NORMALIZE: dict = {
    "brain_mri": "brain_mri",
    "brain_ct": "brain_ct",
    "chest_xray": "chest_xray",
    "histopathology": "histopathology",
    "retina": "retina",
    "skin": "skin",
    "Ultrasound": "ultrasound",
    "ultrasound": "ultrasound",
}

# Modality → internal prefix for image IDs
MODALITY_ABBREVIATION: dict = {
    "brain_mri": "MRI",
    "brain_ct": "CT",
    "chest_xray": "XRAY",
    "histopathology": "HISTO",
    "retina": "RET",
    "skin": "SKIN",
    "ultrasound": "US",
}

# Modality → assumed number of channels (RGB=3, grayscale=1)
MODALITY_CHANNELS: dict = {
    "brain_mri": 1,
    "brain_ct": 1,
    "chest_xray": 1,
    "histopathology": 3,
    "retina": 3,
    "skin": 3,
    "ultrasound": 3,
}

# ---------------------------------------------------------------------------
# Class Normalization Rules
# Each entry is (pattern_list, normalized_name).
# Matching is case-insensitive exact match after stripping and lowercasing.
# ---------------------------------------------------------------------------

CLASS_NORMALIZATION_RULES: list = [
    # Chest X-ray — COVID variants
    (["covid", "covid-19", "covid19", "covid_19"], "covid"),
    # Chest X-ray — Lung Opacity
    (["lung_opacity", "lung opacity", "lungopacity", "lung-opacity"], "lung_opacity"),
    # Chest X-ray — Viral Pneumonia
    (["viral pneumonia", "viral_pneumonia", "viralpneumonia", "viral-pneumonia"], "viral_pneumonia"),
    # Chest X-ray — Normal
    (["normal", "no finding", "nofinding"], "normal"),
    # Brain MRI — Glioma
    (["glioma", "glioma_tumor"], "glioma"),
    # Brain MRI — Meningioma
    (["meningioma", "meningioma_tumor"], "meningioma"),
    # Brain MRI — No Tumor
    (["notumor", "no tumor", "no_tumor", "notumour", "no tumour"], "notumor"),
    # Brain MRI — Pituitary
    (["pituitary", "pituitary_tumor"], "pituitary"),
    # Brain CT
    (["head_ct", "headct", "head ct"], "head_ct"),
    # Histopathology — Colon
    (["colon_aca", "colon aca", "colon adenocarcinoma"], "colon_adenocarcinoma"),
    (["colon_n", "colon n", "colon normal", "colon_normal"], "colon_normal"),
    # Histopathology — Lung
    (["lung_aca", "lung aca", "lung adenocarcinoma"], "lung_adenocarcinoma"),
    (["lung_n", "lung n", "lung normal", "lung_normal"], "lung_normal"),
    (["lung_scc", "lung scc", "lung squamous cell", "lung squamouscell"], "lung_squamous_cell"),
    # Ultrasound — BUSI
    (["benign"], "benign"),
    (["malignant"], "malignant"),
    # Skin — HAM10000 (7 classes)
    (["nv", "melanocytic_nevus", "melanocytic nevi", "melanocytic_nevi", "melanocytic nevus"], "melanocytic_nevus"),
    (["mel", "melanoma"], "melanoma"),
    (["bkl", "benign_keratosis", "benign keratosis", "benign_keratosis_like_lesions", "benign keratosis-like lesions"], "benign_keratosis"),
    (["bcc", "basal_cell_carcinoma", "basal cell carcinoma"], "basal_cell_carcinoma"),
    (["akiec", "actinic_keratosis", "actinic keratoses", "actinic_keratoses", "actinic keratoses / intraepithelial carcinoma"], "actinic_keratosis"),
    (["vasc", "vascular_lesion", "vascular lesions", "vascular_lesions"], "vascular_lesion"),
    (["df", "dermatofibroma"], "dermatofibroma"),
    # Retina / Skin / General — unknown
    (["unknown", "odir-5k", "preprocessed_images", "odir", "retina", "dataset_busi_with_gt",
      "vinbigdata", "lung_colon_image_set"], "unknown"),
]

# ---------------------------------------------------------------------------
# Per-Dataset Class Extraction Strategy
# Keys: (modality_normalized, disease_class_from_module1)
# Values: strategy type and optional parameters
#
# Strategy types:
#   USE_EXISTING   — disease_class from Module 1 is already correct
#   PARENT_FOLDER  — take the parent folder of the image file
#   ANCHOR_OFFSET  — find anchor folder in path, take N levels after it
#   HAM10000_CSV   — match ISIC image ID against HAM10000_metadata.csv
#   UNKNOWN        — cannot extract from path; mark as 'unknown'
# ---------------------------------------------------------------------------

EXTRACTION_STRATEGIES: dict = {
    # Brain MRI — disease_class already correct (glioma, meningioma, notumor, pituitary)
    ("brain_mri", "glioma"): {"strategy": "USE_EXISTING"},
    ("brain_mri", "meningioma"): {"strategy": "USE_EXISTING"},
    ("brain_mri", "notumor"): {"strategy": "USE_EXISTING"},
    ("brain_mri", "pituitary"): {"strategy": "USE_EXISTING"},

    # Brain CT — single class
    ("brain_ct", "head_ct"): {"strategy": "USE_EXISTING"},

    # Chest X-ray — COVID-19 Radiography Dataset
    # Path: .../COVID-19_Radiography_Dataset/{DISEASE}/images/{file}
    ("chest_xray", "COVID-19_Radiography_Dataset"): {
        "strategy": "ANCHOR_OFFSET",
        "anchor": "COVID-19_Radiography_Dataset",
        "offset": 1,
    },

    # Chest X-ray — VinBigData (only splits, no disease class in path)
    ("chest_xray", "vinbigdata"): {"strategy": "UNKNOWN"},

    # Histopathology — Lung & Colon Image Set
    # Path: .../lung_colon_image_set/colon_image_sets/{DISEASE}/{file}
    ("histopathology", "lung_colon_image_set"): {"strategy": "PARENT_FOLDER"},

    # Retina — ODIR-5K (flat images, no disease in path)
    ("retina", "ODIR-5K"): {"strategy": "UNKNOWN"},
    ("retina", "preprocessed_images"): {"strategy": "UNKNOWN"},

    # Skin — HAM10000 (flat images, mapped via HAM10000_metadata.csv)
    ("skin", "HAM10000_images_part_1"): {"strategy": "HAM10000_CSV"},
    ("skin", "HAM10000_images_part_2"): {"strategy": "HAM10000_CSV"},

    # Ultrasound — BUSI
    # Path: .../Dataset_BUSI_with_GT/{DISEASE}/{file}
    ("ultrasound", "Dataset_BUSI_with_GT"): {"strategy": "PARENT_FOLDER"},
}

# ---------------------------------------------------------------------------
# Dataset Registry Info (human-readable metadata per dataset source)
# ---------------------------------------------------------------------------

DATASET_REGISTRY_INFO: dict = {
    "COVID-19_Radiography_Dataset": {
        "name": "COVID-19 Radiography Database",
        "source": "Kaggle / IEEE DataPort",
        "url": "https://www.kaggle.com/tawsifurrahman/covid19-radiography-database",
        "license": "CC BY 4.0",
        "modality": "chest_xray",
        "paper": "Chowdhury et al. (2020) — IEEE Access",
    },
    "vinbigdata": {
        "name": "VinBigData Chest X-ray Abnormalities Detection",
        "source": "Kaggle / VinBigData",
        "url": "https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection",
        "license": "CC BY 4.0",
        "modality": "chest_xray",
        "paper": "VinBigData (2021)",
    },
    "lung_colon_image_set": {
        "name": "Lung and Colon Cancer Histopathological Images (LC25000)",
        "source": "Kaggle / LC25000",
        "url": "https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images",
        "license": "CC BY 4.0",
        "modality": "histopathology",
        "paper": "Borkowski et al. (2019) — arXiv",
    },
    "ODIR-5K": {
        "name": "Ocular Disease Intelligent Recognition (ODIR-5K)",
        "source": "Kaggle / ODIR",
        "url": "https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k",
        "license": "Non-commercial research only",
        "modality": "retina",
        "paper": "ODIR Challenge (2019)",
    },
    "preprocessed_images": {
        "name": "Retina Fundus Images (Preprocessed)",
        "source": "Local / Unknown",
        "url": "",
        "license": "Unknown",
        "modality": "retina",
        "paper": "",
    },
    "HAM10000_images_part_1": {
        "name": "Human Against Machine with 10000 training images (HAM10000)",
        "source": "ISIC Archive / Kaggle",
        "url": "https://www.kaggle.com/kmader/skin-lesion-analysis-toward-melanoma-detection",
        "license": "CC BY-NC 4.0",
        "modality": "skin",
        "paper": "Tschandl et al. (2018) — Nature Scientific Data",
    },
    "HAM10000_images_part_2": {
        "name": "Human Against Machine with 10000 training images (HAM10000)",
        "source": "ISIC Archive / Kaggle",
        "url": "https://www.kaggle.com/kmader/skin-lesion-analysis-toward-melanoma-detection",
        "license": "CC BY-NC 4.0",
        "modality": "skin",
        "paper": "Tschandl et al. (2018) — Nature Scientific Data",
    },
    "Dataset_BUSI_with_GT": {
        "name": "Breast Ultrasound Images Dataset (BUSI) with Ground Truth",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset",
        "license": "CC BY 4.0",
        "modality": "ultrasound",
        "paper": "Al-Dhabyani et al. (2020) — Data in Brief",
    },
    "head_ct": {
        "name": "Head CT Scan Dataset",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/felipekitamura/head-ct-hemorrhage",
        "license": "Unknown",
        "modality": "brain_ct",
        "paper": "",
    },
}

# ---------------------------------------------------------------------------
# V2.0 Canonical Metadata Schema & Manifests Configuration
# ---------------------------------------------------------------------------

MANIFESTS_DIR: Path = PROJECT_ROOT / "data" / "manifests"

COL_V2_IMAGE_ID = "image_id"
COL_V2_IMAGE_PATH = "image_path"
COL_V2_MODALITY = "modality"
COL_V2_DATASET_NAME = "dataset_name"
COL_V2_DATASET_VERSION = "dataset_version"
COL_V2_SOURCE = "source"
COL_V2_ORIGINAL_LABEL = "original_label"
COL_V2_STANDARDIZED_LABEL = "standardized_label"
COL_V2_LABEL_TYPE = "label_type"
COL_V2_CLASS_ID = "class_id"
COL_V2_MULTI_LABEL_VECTOR = "multi_label_vector"
COL_V2_PATIENT_ID = "patient_id"
COL_V2_STUDY_ID = "study_id"
COL_V2_LESION_ID = "lesion_id"
COL_V2_EYE_ID = "eye_id"
COL_V2_LEAKAGE_GROUP_ID = "leakage_group_id"
COL_V2_LABEL_STATUS = "label_status"
COL_V2_METADATA_SOURCE = "metadata_source"
COL_V2_MASK_STATUS = "mask_status"
COL_V2_QUALITY_STATUS = "quality_status"
COL_V2_SPLIT_STATUS = "split_status"
COL_V2_DUPLICATE_STATUS = "duplicate_status"

V2_CANONICAL_COLUMNS = [
    COL_V2_IMAGE_ID,
    COL_V2_IMAGE_PATH,
    COL_V2_MODALITY,
    COL_V2_DATASET_NAME,
    COL_V2_DATASET_VERSION,
    COL_V2_SOURCE,
    COL_V2_ORIGINAL_LABEL,
    COL_V2_STANDARDIZED_LABEL,
    COL_V2_LABEL_TYPE,
    COL_V2_CLASS_ID,
    COL_V2_MULTI_LABEL_VECTOR,
    COL_V2_PATIENT_ID,
    COL_V2_STUDY_ID,
    COL_V2_LESION_ID,
    COL_V2_EYE_ID,
    COL_V2_LEAKAGE_GROUP_ID,
    COL_V2_LABEL_STATUS,
    COL_V2_METADATA_SOURCE,
    COL_V2_MASK_STATUS,
    COL_V2_QUALITY_STATUS,
    COL_V2_SPLIT_STATUS,
    COL_V2_DUPLICATE_STATUS,
]

# Modality dataset versions
DATASET_VERSIONS: dict = {
    "brain_mri": "brain_mri_v2.0",
    "brain_ct": "brain_ct_v2.0",
    "chest_xray": "chest_xray_v2.0",
    "histopathology": "histopathology_v2.0",
    "retina": "retina_v2.0",
    "skin": "skin_v2.0",
    "ultrasound": "ultrasound_v2.0",
}

# Modality standardized classes and class IDs
V2_MODALITY_CLASSES: dict = {
    "brain_mri": {
        "glioma": 0,
        "meningioma": 1,
        "no_tumor": 2,
        "pituitary": 3,
    },
    "brain_ct": {
        "normal": 0,
        "hemorrhage": 1,
    },
    "chest_xray": {
        "covid": 0,
        "lung_opacity": 1,
        "normal": 2,
        "viral_pneumonia": 3,
    },
    "histopathology": {
        "colon_adenocarcinoma": 0,
        "colon_normal": 1,
        "lung_adenocarcinoma": 2,
        "lung_normal": 3,
        "lung_squamous_cell_carcinoma": 4,
    },
    "retina": {
        "normal": 0,
        "diabetic_retinopathy": 1,
        "glaucoma": 2,
        "cataract": 3,
        "amd": 4,
        "hypertension": 5,
        "pathological_myopia": 6,
        "other_abnormality": 7,
    },
    "skin": {
        "melanocytic_nevus": 0,
        "melanoma": 1,
        "benign_keratosis": 2,
        "basal_cell_carcinoma": 3,
        "actinic_keratosis": 4,
        "vascular_lesion": 5,
        "dermatofibroma": 6,
    },
    "ultrasound": {
        "benign": 0,
        "malignant": 1,
        "normal": 2,
    },
}
