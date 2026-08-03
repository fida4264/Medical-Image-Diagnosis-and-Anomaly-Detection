"""
eda/config.py
==============
Centralised configuration for Module 5: Medical Dataset EDA.

All paths, column names, supported modalities, and output file names are
defined here so every sub-module imports from a single source of truth.

Author: Medical AI System — Module 5
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Input paths — outputs from previous modules
# ---------------------------------------------------------------------------

# Module 1 outputs
MODULE1_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LABELS_CSV: Path = MODULE1_OUTPUTS_DIR / "labels.csv"
CLASS_DISTRIBUTION_CSV: Path = MODULE1_OUTPUTS_DIR / "class_distribution.csv"

# Module 2 outputs
METADATA_DIR: Path = PROJECT_ROOT / "metadata"
MASTER_METADATA_CSV: Path = METADATA_DIR / "master_metadata.csv"
DATASET_REGISTRY_JSON: Path = METADATA_DIR / "dataset_registry.json"

# Module 3 outputs
PROCESSED_IMAGES_DIR: Path = PROJECT_ROOT / "processed_images"
PREPROCESSING_METADATA_CSV: Path = PROCESSED_IMAGES_DIR / "preprocessing_metadata.csv"
PREPROCESSING_REPORT_JSON: Path = PROCESSED_IMAGES_DIR / "preprocessing_report.json"
QUALITY_REPORT_JSON: Path = PROCESSED_IMAGES_DIR / "quality_report.json"

# Module 4 outputs
AUGMENTED_IMAGES_DIR: Path = PROJECT_ROOT / "augmented_images"
AUGMENTATION_METADATA_CSV: Path = AUGMENTED_IMAGES_DIR / "augmentation_metadata.csv"
AUGMENTATION_REPORT_JSON: Path = AUGMENTED_IMAGES_DIR / "augmentation_report.json"
AUGMENTATION_STATISTICS_JSON: Path = AUGMENTED_IMAGES_DIR / "augmentation_statistics.json"

# ---------------------------------------------------------------------------
# EDA output directory
# ---------------------------------------------------------------------------

EDA_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "eda"
EDA_VISUALIZATIONS_DIR: Path = EDA_OUTPUT_DIR / "visualizations"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# EDA configuration
EDA_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "eda.yaml"

# Output file names
EDA_STATISTICS_JSON: str = "eda_statistics.json"
EDA_SUMMARY_MD: str = "eda_summary.md"
EDA_REPORT_PDF: str = "eda_report.pdf"
EDA_TABLES_CSV: str = "eda_tables.csv"
VISUALIZATION_INDEX_MD: str = "visualization_index.md"
DATASET_QUALITY_SCORE_JSON: str = "dataset_quality_score.json"
LOG_FILE_NAME: str = "eda.log"

# ---------------------------------------------------------------------------
# Supported modalities — must match Modules 2–4
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES: list = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "retina",
    "skin",
    "ultrasound",
    "histopathology",
]

MODALITY_DISPLAY_NAMES: dict = {
    "brain_mri":      "Brain MRI",
    "brain_ct":       "Brain CT",
    "chest_xray":     "Chest X-Ray",
    "retina":         "Retina",
    "skin":           "Skin Lesion",
    "ultrasound":     "Ultrasound",
    "histopathology": "Histopathology",
}

# Grayscale modalities (for colour analysis routing)
GRAYSCALE_MODALITIES: set = {"brain_mri", "brain_ct", "chest_xray"}
RGB_MODALITIES: set = {"retina", "skin", "histopathology", "ultrasound"}

# ---------------------------------------------------------------------------
# Column names — master_metadata.csv
# ---------------------------------------------------------------------------

COL_IMAGE_ID         = "image_id"
COL_INTERNAL_ID      = "internal_id"
COL_DATASET_NAME     = "dataset_name"
COL_DATASET_SOURCE   = "dataset_source"
COL_MODALITY         = "modality"
COL_DISEASE          = "disease"
COL_NORMALIZED_CLASS = "normalized_class"
COL_ORIGINAL_CLASS   = "original_class"
COL_SPLIT            = "split"
COL_IMAGE_PATH       = "image_path"
COL_EXTENSION        = "extension"
COL_WIDTH            = "width"
COL_HEIGHT           = "height"
COL_CHANNELS         = "channels"
COL_FILE_SIZE        = "file_size_bytes"

# Column names — preprocessing_metadata.csv
COL_ORIGINAL_WIDTH   = "original_width"
COL_ORIGINAL_HEIGHT  = "original_height"
COL_PROCESSED_WIDTH  = "processed_width"
COL_PROCESSED_HEIGHT = "processed_height"
COL_OPERATIONS       = "operations_applied"
COL_PROC_TIME        = "processing_time_s"
COL_STATUS           = "status"
COL_QUALITY_FLAGS    = "quality_flags"
COL_PROCESSED_PATH   = "processed_path"

# Column names — augmentation_metadata.csv
COL_AUG_ORIGINAL_ID  = "original_image_id"
COL_AUG_INDEX        = "augmentation_index"
COL_AUG_OPS          = "operations_applied"
COL_AUG_SEED         = "seed"
COL_AUG_STATUS       = "status"
COL_AUG_PATH         = "augmented_path"
