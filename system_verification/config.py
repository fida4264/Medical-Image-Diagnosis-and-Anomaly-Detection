"""
system_verification/config.py
===============================
Centralised configuration for Module 5.5: System Verification & Integration Testing.

All expected paths, column names, thresholds, scoring weights, and known
constants are defined here as a single source of truth.

Author: Medical AI System — Module 5.5
"""

from pathlib import Path
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Validation Output Directory
# ---------------------------------------------------------------------------

VALIDATION_DIR: Path = PROJECT_ROOT / "validation"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE_NAME: str = "system_verification.log"

# ---------------------------------------------------------------------------
# Expected Project Structure
# ---------------------------------------------------------------------------

EXPECTED_PACKAGES: List[str] = [
    "dataset_manager",
    "metadata_manager",
    "preprocessing",
    "augmentation",
    "eda",
    "system_verification",
    "tests",
]

EXPECTED_CONFIGS: List[str] = [
    "configs/preprocessing.yaml",
    "configs/augmentation.yaml",
    "configs/eda.yaml",
]

EXPECTED_ENTRY_POINTS: List[str] = [
    "run_dataset_manager.py",
    "run_metadata_manager.py",
    "run_preprocessing.py",
    "run_augmentation.py",
    "run_eda.py",
    "run_verification.py",
]

EXPECTED_DOCS: List[str] = [
    "docs/module1_design.md",
    "docs/module2_metadata_management.md",
    "docs/module3_preprocessing.md",
]

EXPECTED_ROOT_FILES: List[str] = [
    "requirements.txt",
    "README.md",
    "VERSION.json",
    "PROJECT_STATE.md",
]

# ---------------------------------------------------------------------------
# Module 1 Expected Outputs
# ---------------------------------------------------------------------------

MODULE1_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
M1_METADATA_CSV: Path = MODULE1_OUTPUTS_DIR / "metadata.csv"
M1_LABELS_CSV: Path = MODULE1_OUTPUTS_DIR / "labels.csv"
M1_STATISTICS_JSON: Path = MODULE1_OUTPUTS_DIR / "dataset_statistics.json"
M1_VALIDATION_REPORT_JSON: Path = MODULE1_OUTPUTS_DIR / "validation_report.json"
M1_CLASS_DISTRIBUTION_CSV: Path = MODULE1_OUTPUTS_DIR / "class_distribution.csv"

M1_EXPECTED_FILES: List[Path] = [
    M1_METADATA_CSV,
    M1_LABELS_CSV,
    M1_STATISTICS_JSON,
    M1_VALIDATION_REPORT_JSON,
    M1_CLASS_DISTRIBUTION_CSV,
]

# ---------------------------------------------------------------------------
# Module 2 Expected Outputs
# ---------------------------------------------------------------------------

MODULE2_OUTPUTS_DIR: Path = PROJECT_ROOT / "metadata"
M2_MASTER_METADATA_CSV: Path = MODULE2_OUTPUTS_DIR / "master_metadata.csv"
M2_CLASS_MAPPING_CSV: Path = MODULE2_OUTPUTS_DIR / "class_mapping.csv"
M2_IMAGE_MAPPING_CSV: Path = MODULE2_OUTPUTS_DIR / "image_mapping.csv"
M2_MODALITY_MAPPING_CSV: Path = MODULE2_OUTPUTS_DIR / "modality_mapping.csv"
M2_DATASET_REGISTRY_JSON: Path = MODULE2_OUTPUTS_DIR / "dataset_registry.json"
M2_DATASET_SUMMARY_MD: Path = MODULE2_OUTPUTS_DIR / "dataset_summary.md"
M2_VISUALIZATIONS_DIR: Path = MODULE2_OUTPUTS_DIR / "visualizations"

M2_EXPECTED_FILES: List[Path] = [
    M2_MASTER_METADATA_CSV,
    M2_CLASS_MAPPING_CSV,
    M2_IMAGE_MAPPING_CSV,
    M2_MODALITY_MAPPING_CSV,
    M2_DATASET_REGISTRY_JSON,
    M2_DATASET_SUMMARY_MD,
]

M2_EXPECTED_VISUALIZATIONS: List[str] = [
    "class_distribution_bar.png",
    "dataset_source_bar.png",
    "modality_class_heatmap.png",
    "modality_distribution_pie.png",
    "resolution_scatter.png",
]

# ---------------------------------------------------------------------------
# Module 3 Expected Outputs
# ---------------------------------------------------------------------------

MODULE3_OUTPUTS_DIR: Path = PROJECT_ROOT / "processed_images"
M3_PREPROCESSING_METADATA_CSV: Path = MODULE3_OUTPUTS_DIR / "preprocessing_metadata.csv"
M3_PREPROCESSING_REPORT_JSON: Path = MODULE3_OUTPUTS_DIR / "preprocessing_report.json"
M3_QUALITY_REPORT_JSON: Path = MODULE3_OUTPUTS_DIR / "quality_report.json"

M3_EXPECTED_FILES: List[Path] = [
    M3_PREPROCESSING_METADATA_CSV,
    M3_PREPROCESSING_REPORT_JSON,
    M3_QUALITY_REPORT_JSON,
]

M3_EXPECTED_MODALITY_DIRS: List[str] = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "histopathology",
    "retina",
    "skin",
    "ultrasound",
]

# ---------------------------------------------------------------------------
# Module 4 Expected Outputs
# ---------------------------------------------------------------------------

MODULE4_OUTPUTS_DIR: Path = PROJECT_ROOT / "augmented_images"
M4_AUGMENTATION_METADATA_CSV: Path = MODULE4_OUTPUTS_DIR / "augmentation_metadata.csv"
M4_AUGMENTATION_REPORT_JSON: Path = MODULE4_OUTPUTS_DIR / "augmentation_report.json"
M4_AUGMENTATION_STATISTICS_JSON: Path = MODULE4_OUTPUTS_DIR / "augmentation_statistics.json"

M4_EXPECTED_FILES: List[Path] = [
    M4_AUGMENTATION_METADATA_CSV,
    M4_AUGMENTATION_REPORT_JSON,
    M4_AUGMENTATION_STATISTICS_JSON,
]

# ---------------------------------------------------------------------------
# Module 5 Expected Outputs
# ---------------------------------------------------------------------------

MODULE5_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs" / "eda"
M5_EDA_STATISTICS_JSON: Path = MODULE5_OUTPUTS_DIR / "eda_statistics.json"
M5_EDA_SUMMARY_MD: Path = MODULE5_OUTPUTS_DIR / "eda_summary.md"
M5_EDA_TABLES_CSV: Path = MODULE5_OUTPUTS_DIR / "eda_tables.csv"
M5_DATASET_QUALITY_SCORE_JSON: Path = MODULE5_OUTPUTS_DIR / "dataset_quality_score.json"
M5_VISUALIZATION_INDEX_MD: Path = MODULE5_OUTPUTS_DIR / "visualization_index.md"
M5_VISUALIZATIONS_DIR: Path = MODULE5_OUTPUTS_DIR / "visualizations"

M5_EXPECTED_FILES: List[Path] = [
    M5_EDA_STATISTICS_JSON,
    M5_EDA_SUMMARY_MD,
    M5_EDA_TABLES_CSV,
    M5_DATASET_QUALITY_SCORE_JSON,
    M5_VISUALIZATION_INDEX_MD,
]

# ---------------------------------------------------------------------------
# Log Files
# ---------------------------------------------------------------------------

EXPECTED_LOG_FILES: List[str] = [
    "dataset_manager.log",
    "metadata_manager.log",
    "preprocessing.log",
]

# ---------------------------------------------------------------------------
# master_metadata.csv Schema
# ---------------------------------------------------------------------------

MASTER_METADATA_REQUIRED_COLUMNS: List[str] = [
    "image_id",
    "internal_id",
    "dataset_name",
    "dataset_source",
    "modality",
    "disease",
    "normalized_class",
    "original_class",
    "split",
    "image_path",
    "extension",
    "width",
    "height",
    "channels",
    "file_size_bytes",
]

SUPPORTED_MODALITIES: Set[str] = {
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "histopathology",
    "retina",
    "skin",
    "ultrasound",
}

SUPPORTED_SPLITS: Set[str] = {"train", "val", "test", "unspecified"}

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Required Packages
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
    "albumentations": "albumentations",
    "scipy": "scipy",
}

# ---------------------------------------------------------------------------
# Health Scoring Weights
# ---------------------------------------------------------------------------

HEALTH_WEIGHTS: Dict[str, float] = {
    "architecture":         10.0,
    "data_pipeline":        10.0,
    "metadata":              8.0,
    "preprocessing":         8.0,
    "augmentation":          8.0,
    "eda":                   7.0,
    "configuration":         7.0,
    "documentation":         7.0,
    "testing":              10.0,
    "maintainability":       5.0,
    "scalability":           5.0,
    "performance":           5.0,
    "research_readiness":    5.0,
    "deployment_readiness":  5.0,
}

# GO / NO-GO threshold (0–100)
GO_THRESHOLD: float = 70.0

# ---------------------------------------------------------------------------
# Validation Output Report Files
# ---------------------------------------------------------------------------

REPORT_FILES: Dict[str, str] = {
    "project_validation":    "project_validation.md",
    "integration_report":    "integration_report.md",
    "health_score":          "project_health_score.json",
    "performance_report":    "performance_report.md",
    "code_review":           "code_review.md",
    "dependency_report":     "dependency_report.md",
    "module_status":         "module_status.json",
    "project_readiness":     "project_readiness.json",
    "architecture_validation": "architecture_validation.md",
    "pipeline_validation":   "pipeline_validation.md",
    "dashboard":             "validation_dashboard.html",
}

# ---------------------------------------------------------------------------
# Performance Thresholds
# ---------------------------------------------------------------------------

PERF_CSV_LOAD_WARNING_S: float = 10.0    # Warn if CSV loads slower than this
PERF_IMAGE_LOAD_WARNING_S: float = 0.5   # Warn if single image loads slower
PERF_MEMORY_WARNING_MB: float = 500.0    # Warn if DataFrame exceeds this size

# ---------------------------------------------------------------------------
# Sampling Constants
# ---------------------------------------------------------------------------

PATH_SAMPLE_SIZE: int = 50    # Number of image paths to spot-check from metadata
