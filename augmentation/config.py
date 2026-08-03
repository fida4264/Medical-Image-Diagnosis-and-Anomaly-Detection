"""
augmentation/config.py
=======================
Centralised constants and path resolution for Module 4:
Medical Image Augmentation Framework.

Mirrors metadata_manager/config.py and preprocessing/ patterns.
All downstream modules import paths from here — no hard-coded strings.

Author: Medical AI System — Module 4
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Root & Directory Layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Module 3 outputs (inputs to this module)
PROCESSED_IMAGES_DIR: Path = PROJECT_ROOT / "processed_images"
PREPROCESSING_METADATA_CSV: Path = PROCESSED_IMAGES_DIR / "preprocessing_metadata.csv"
PREPROCESSING_REPORT_JSON: Path = PROCESSED_IMAGES_DIR / "preprocessing_report.json"

# Module 2 outputs (fallback metadata)
METADATA_DIR: Path = PROJECT_ROOT / "metadata"
MASTER_METADATA_CSV: Path = METADATA_DIR / "master_metadata.csv"
DATASET_REGISTRY_JSON: Path = METADATA_DIR / "dataset_registry.json"

# Module 4 outputs
AUGMENTED_IMAGES_DIR: Path = PROJECT_ROOT / "augmented_images"
AUGMENTATION_OUTPUT_DIR: Path = PROJECT_ROOT / "augmentation"  # reports live here
AUGMENTATION_PREVIEW_DIR: Path = PROJECT_ROOT / "augmentation_preview"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Configuration
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "augmentation.yaml"

# Output file names
AUGMENTATION_METADATA_CSV: str = "augmentation_metadata.csv"
AUGMENTATION_REPORT_JSON: str = "augmentation_report.json"
AUGMENTATION_STATISTICS_JSON: str = "augmentation_statistics.json"
LOG_FILE_NAME: str = "augmentation.log"

# ---------------------------------------------------------------------------
# Supported Modalities (must match Module 3)
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

# ---------------------------------------------------------------------------
# Augmentation Metadata CSV columns
# ---------------------------------------------------------------------------

AUGMENTATION_METADATA_COLUMNS: list = [
    "original_image_id",
    "original_path",
    "augmented_path",
    "modality",
    "normalized_class",
    "augmentation_index",
    "operations_applied",
    "seed",
    "timestamp",
    "status",
]
