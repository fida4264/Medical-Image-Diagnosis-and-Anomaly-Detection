# Module 1: Dataset Management — Design Document

## Overview

Module 1 is the **foundation layer** of the Medical Image Diagnosis and
Anomaly Detection System.  Every other module in the project (2–24) depends
on the outputs and classes defined here.

---

## Responsibilities

| Responsibility | Implementation |
|---|---|
| Discover all datasets recursively | `DatasetScanner` |
| Detect modalities automatically | `DatasetScanner._get_modality_directories()` |
| Detect disease class folders | `DatasetScanner._scan_layout_ab()` |
| Detect train/val/test splits | `DatasetScanner._scan_layout_c()`, `_all_are_splits()` |
| Collect every image path | `DatasetScanner._collect_images()` |
| Support jpg/jpeg/png/bmp/tif/tiff | `config.SUPPORTED_EXTENSIONS` |
| Ignore hidden files | `image_utils.is_hidden_file()` |
| Ignore unsupported files | `image_utils.is_supported_image()` |
| Validate integrity | `DatasetValidator` |
| Generate `metadata.csv` | `MetadataGenerator` |
| Generate `labels.csv` | `LabelGenerator` |
| Generate `dataset_statistics.json` | `StatisticsGenerator` |
| Generate `validation_report.json` | `DatasetValidator.save_report()` |
| Generate `class_distribution.csv` | `LabelGenerator.save_class_distribution()` |

---

## Supported Dataset Layouts

The scanner auto-detects which layout is in use for each modality. All three
layouts can coexist across different modality folders in the same dataset root.

### Layout A — Flat Disease Folders
```
datasets/
    chest_xray/
        Pneumonia/
            img001.jpg
        Normal/
            img002.jpg
```
Split column: `"unspecified"`

### Layout B — Disease → Split Sub-Folders
```
datasets/
    brain_mri/
        Glioma/
            train/
                img001.jpg
            test/
                img002.jpg
```
Split column: `"train"` / `"test"` / `"val"`

### Layout C — Split → Disease Sub-Folders
```
datasets/
    skin/
        train/
            Melanoma/
                img001.jpg
        val/
            Melanoma/
                img002.jpg
```
Split column: `"train"` / `"val"`

---

## Output File Schemas

### metadata.csv

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Unique ID (relative path, forward-slashed) |
| `modality` | str | Imaging modality folder name |
| `disease_class` | str | Disease class folder name |
| `split` | str | `"train"` / `"val"` / `"test"` / `"unspecified"` |
| `image_path` | str | Absolute path to image file |
| `filename` | str | Bare filename with extension |
| `extension` | str | Lowercase extension with dot |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `file_size_bytes` | int | File size in bytes |
| `is_valid` | bool | True if PIL can fully decode the image |
| `md5_hash` | str | Hex MD5 digest |

### labels.csv

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Foreign key → metadata.csv |
| `modality` | str | Imaging modality |
| `disease_class` | str | Raw class folder name |
| `split` | str | Dataset split |
| `label` | int | Zero-based integer label (per modality) |
| `class_name` | str | Canonical class name |

### class_distribution.csv

| Column | Type | Description |
|---|---|---|
| `modality` | str | Imaging modality |
| `disease_class` | str | Disease class |
| `split` | str | Dataset split |
| `count` | int | Number of valid images |

### dataset_statistics.json

Hierarchical JSON:
```
{
  "generated_at": "ISO-8601",
  "overall": {
    "total_images": int,
    "total_valid_images": int,
    "total_modalities": int,
    "total_classes": int,
    ...
  },
  "per_modality": {
    "brain_mri": {
      "total_images": int,
      "num_classes": int,
      "avg_width_px": float,
      "per_class": {"Glioma": 100, "Normal": 150},
      "per_split": {"train": 200, "test": 50}
    },
    ...
  }
}
```

### validation_report.json

```
{
  "total_images_scanned": int,
  "errors": int,
  "warnings": int,
  "summary": {"CORRUPTED": 2, "DUPLICATE": 1},
  "duplicate_groups": {"<md5>": ["path1", "path2"]},
  "issues": [
    {
      "severity": "ERROR",
      "category": "CORRUPTED",
      "description": "...",
      "affected_path": "...",
      "modality": "brain_mri",
      "disease_class": "Glioma"
    },
    ...
  ]
}
```

---

## Architecture Decisions

### Zero Hardcoding
No disease class or modality name is hardcoded anywhere in the codebase.
`KNOWN_MODALITIES` in `config.py` is used only to warn about expected folders
that are absent — it never controls which folders are scanned.

### ImageRecord as Frozen Dataclass
`ImageRecord` is a frozen dataclass. This guarantees that once the scanner
produces a record, no downstream code can accidentally mutate it, preventing
hard-to-debug state bugs.

### Per-Modality Label Namespaces
Integer labels are scoped per modality. `"Normal"` in `brain_mri` and
`"Normal"` in `chest_xray` are assigned the same integer value (0, 1, or 2
depending on alphabetical order) independently. This prevents label leakage
between modalities and allows each specialist model to use its own label space.

### MD5 Hashing for Deduplication
MD5 is used for its speed. This is a data engineering context, not a security
context, so MD5 is appropriate. All hashing is done in 64 KB chunks to
handle very large medical images without loading them entirely into RAM.

### Lazy Image Header Reading
`get_image_dimensions()` uses `PIL.Image.open()` without calling `.load()`,
which only reads the image header. This is 10–100× faster than full decoding
for large TIFF/BMP files and is sufficient for dimension extraction.

---

## Dependencies

| Package | Use |
|---|---|
| `Pillow` | Image loading, dimension reading, corruption detection |
| `pandas` | DataFrame construction and CSV I/O |
| `pathlib` | All path operations (no string path concatenation) |
| `hashlib` | MD5 computation |
| `logging` | Structured, levelled logging |
| `dataclasses` | `ImageRecord`, `ValidationIssue`, `ValidationReport` |

---

## How Future Modules Use Module 1

```python
# Module 4 — Preprocessing
from dataset_manager import DatasetScanner, ImageRecord
import pandas as pd

meta = pd.read_csv("outputs/metadata.csv")
valid_records = meta[meta["is_valid"]]
# ... apply preprocessing ...

# Module 8 — Brain MRI Model
meta = pd.read_csv("outputs/metadata.csv")
labels = pd.read_csv("outputs/labels.csv")
brain_data = meta[meta["modality"] == "brain_mri"]
# No scanning needed — use the pre-built tables.
```
