"""
config.py
=========
Central configuration module for the Medical Image Diagnosis and Anomaly Detection System.

All project-wide constants, paths, and settings are defined here.
Every module in this project must import configuration values from this file —
never hardcode paths or settings in individual modules.

This design guarantees that the entire system is reconfigured by changing only this file.

Author: Medical AI System — Module 1
"""

from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------

# Root directory of the entire project (parent of dataset_manager/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Root directory where all modality dataset folders live
DATASETS_ROOT: Path = PROJECT_ROOT / "datasets"

# Directory where all generated output files (CSV, JSON) will be written
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Directory where runtime logs are written
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Supported Image File Extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
})

# ---------------------------------------------------------------------------
# Known Image Modalities
# ---------------------------------------------------------------------------
# These are the EXPECTED sub-folder names inside DATASETS_ROOT.
# The scanner will auto-discover ANY folder it finds — this list is only used
# for validation warnings (to flag unexpected or missing modality folders).
# New modalities can be added here without modifying any other module.

KNOWN_MODALITIES: tuple = (
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "retina",
    "skin",
    "ultrasound",
    "histopathology",
)

# ---------------------------------------------------------------------------
# Split Folder Detection
# ---------------------------------------------------------------------------
# When a disease folder contains sub-folders matching these keywords,
# the scanner treats them as dataset splits rather than nested disease classes.
# Comparison is case-insensitive.

SPLIT_KEYWORDS: frozenset = frozenset({
    "train",
    "training",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
})

# ---------------------------------------------------------------------------
# Output File Names
# ---------------------------------------------------------------------------

METADATA_CSV: str = "metadata.csv"
LABELS_CSV: str = "labels.csv"
STATISTICS_JSON: str = "dataset_statistics.json"
VALIDATION_REPORT_JSON: str = "validation_report.json"
CLASS_DISTRIBUTION_CSV: str = "class_distribution.csv"

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL: int = logging.DEBUG
LOG_FORMAT: str = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME: str = "dataset_manager.log"

# ---------------------------------------------------------------------------
# Image Validation Thresholds
# ---------------------------------------------------------------------------

# Images smaller than this (in bytes) are flagged as likely corrupted/empty
MIN_FILE_SIZE_BYTES: int = 1_024        # 1 KB

# Images larger than this are flagged with a warning (not an error)
MAX_FILE_SIZE_BYTES: int = 50_000_000   # 50 MB

# Minimum acceptable image dimension in pixels (width or height)
MIN_IMAGE_DIMENSION: int = 10

# ---------------------------------------------------------------------------
# Metadata Column Names
# ---------------------------------------------------------------------------
# Centralising column names so all modules share the same schema.

COL_IMAGE_ID: str = "image_id"
COL_MODALITY: str = "modality"
COL_DISEASE_CLASS: str = "disease_class"
COL_SPLIT: str = "split"
COL_IMAGE_PATH: str = "image_path"
COL_FILENAME: str = "filename"
COL_EXTENSION: str = "extension"
COL_WIDTH: str = "width"
COL_HEIGHT: str = "height"
COL_FILE_SIZE_BYTES: str = "file_size_bytes"
COL_IS_VALID: str = "is_valid"
COL_MD5_HASH: str = "md5_hash"
COL_LABEL: str = "label"
COL_CLASS_NAME: str = "class_name"
COL_COUNT: str = "count"
