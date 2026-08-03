# Module 2: Dataset Standardization & Metadata Management

## Overview

Module 2 consumes the five CSV/JSON files produced by Module 1 and transforms
them into a unified internal representation that all downstream modules (3–24)
can use without knowing the original dataset structure.

**No original files are modified.** Everything is done through metadata mappings.

---

## Architecture

```
metadata_manager/
├── config.py                  — All paths, constants, normalization rules
├── __init__.py                — Public API
├── metadata_loader.py         — Reads Module 1 outputs, validates schema
├── metadata_standardizer.py   — Core: normalize, extract, align schema
├── class_mapper.py            — Rule-based class name normalization
├── image_mapper.py            — Unique internal ID generation
├── registry_generator.py      — Per-dataset provenance catalogue
├── summary_generator.py       — Human-readable Markdown report
└── visualization.py           — 5 publication-quality charts
```

---

## Inputs (from Module 1)

| File | Source Module | Description |
|---|---|---|
| `outputs/metadata.csv` | Module 1 | 118,715 rows — raw per-image metadata |
| `outputs/labels.csv` | Module 1 | Integer label per image |
| `outputs/class_distribution.csv` | Module 1 | Count per modality/class/split |
| `outputs/dataset_statistics.json` | Module 1 | Aggregated numerical statistics |
| `outputs/validation_report.json` | Module 1 | Errors and warnings from validation |

---

## Outputs

| File | Description |
|---|---|
| `metadata/master_metadata.csv` | 118,715 rows × 15 cols — unified schema |
| `metadata/image_mapping.csv` | original path ↔ internal ID |
| `metadata/class_mapping.csv` | raw class → normalized class |
| `metadata/modality_mapping.csv` | raw modality → abbreviation |
| `metadata/dataset_registry.json` | per-dataset provenance catalogue |
| `metadata/dataset_summary.md` | human-readable statistics report |
| `metadata/visualizations/*.png` | 5 charts |

---

## master_metadata.csv Schema

| Column | Type | Description |
|---|---|---|
| `image_id` | str | Original Module 1 composite ID |
| `internal_id` | str | Unique system ID (e.g., `MRI_000001`) |
| `dataset_name` | str | Human-readable dataset name |
| `dataset_source` | str | Platform/URL source |
| `modality` | str | Normalised modality name |
| `disease` | str | Raw path-extracted disease class |
| `normalized_class` | str | Canonical lowercase class name |
| `original_class` | str | Raw class before normalization |
| `split` | str | `train` / `test` / `unspecified` |
| `image_path` | str | Absolute path to original file |
| `extension` | str | File extension (`.jpg`, `.png`, …) |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `channels` | int | 1 = grayscale, 3 = RGB |
| `file_size_bytes` | int | File size in bytes |

---

## Class Normalization

Class names are normalized deterministically using lookup rules in `config.py`.

### Examples

| Raw Class | Normalized Class |
|---|---|
| `COVID`, `COVID-19`, `Covid19` | `covid` |
| `Lung Opacity`, `Lung_Opacity` | `lung_opacity` |
| `Viral Pneumonia` | `viral_pneumonia` |
| `Normal`, `NORMAL` | `normal` |
| `No Tumor`, `NoTumor`, `No_Tumor` | `notumor` |
| `glioma` | `glioma` |
| `meningioma` | `meningioma` |
| `pituitary` | `pituitary` |
| `colon_aca` | `colon_adenocarcinoma` |
| `colon_n` | `colon_normal` |
| `lung_aca` | `lung_adenocarcinoma` |
| `lung_n` | `lung_normal` |
| `lung_scc` | `lung_squamous_cell` |
| `benign` | `benign` |
| `malignant` | `malignant` |
| Unrecognised | `unknown` |

---

## Disease Class Extraction Strategy

Module 1's `disease_class` field sometimes contains the **dataset folder name**
rather than the actual disease (e.g., `COVID-19_Radiography_Dataset`).
The standardizer extracts the true disease class from the image path using
dataset-specific strategies:

| Dataset | Strategy | Example |
|---|---|---|
| `brain_mri` | USE_EXISTING | `glioma` (already correct) |
| `brain_ct` | USE_EXISTING | `head_ct` |
| `COVID-19_Radiography_Dataset` | ANCHOR_OFFSET | path[anchor+1] → `COVID` |
| `lung_colon_image_set` | PARENT_FOLDER | parent dir → `colon_aca` |
| `Dataset_BUSI_with_GT` | PARENT_FOLDER | parent dir → `benign` |
| `vinbigdata`, `ODIR-5K`, `HAM10000` | UNKNOWN | `unknown` |

---

## Modality Mapping

| Raw Folder | Normalized | Abbreviation |
|---|---|---|
| `brain_mri` | `brain_mri` | `MRI` |
| `brain_ct` | `brain_ct` | `CT` |
| `chest_xray` | `chest_xray` | `XRAY` |
| `histopathology` | `histopathology` | `HISTO` |
| `retina` | `retina` | `RET` |
| `skin` | `skin` | `SKIN` |
| `Ultrasound` | `ultrasound` | `US` |

---

## Internal Image ID Format

```
{MODALITY_PREFIX}_{SEQUENCE:06d}
```

Examples:
- `MRI_000001`, `MRI_000002`, …
- `XRAY_000001`, `XRAY_060330`
- `HISTO_000001`, `HISTO_025000`

IDs are globally unique across all modalities and stable within a run.

---

## Pipeline Workflow

```
python run_metadata_manager.py
```

```
[1/6] Load       ← MetadataLoader reads 5 files from outputs/
[2/6] Standardize ← MetadataStandardizer: normalize + extract + align schema
[3/6] Map Classes ← ClassMapper: class_mapping.csv + modality_mapping.csv
[4/6] Map Images  ← ImageMapper: image_mapping.csv + assign internal IDs
[5/6] Registry   ← RegistryGenerator: dataset_registry.json
[6/6] Summarise  ← SummaryGenerator + Visualizer: summary.md + 5 charts
```

Expected runtime: **~15–30 seconds** for 118,715 images.

---

## Usage by Downstream Modules

```python
import pandas as pd

# Load master metadata
master = pd.read_csv("metadata/master_metadata.csv")

# Filter by modality
brain_mri = master[master["modality"] == "brain_mri"]

# Filter by normalized class
covid_images = master[master["normalized_class"] == "covid"]

# Look up original path from internal ID
row = master[master["internal_id"] == "MRI_000042"].iloc[0]
original_path = row["image_path"]
```

---

## Logs

All operations are logged to:
```
logs/metadata_manager.log
```

Log level: `INFO` (file) and `INFO` (console).

---

## Tests

```bash
# Run all Module 2 tests
python -m pytest tests/test_metadata_loader.py tests/test_class_mapper.py \
       tests/test_image_mapper.py tests/test_registry_generator.py \
       tests/test_summary_generator.py -v
```

| Test File | Classes Tested | Test Count |
|---|---|---|
| `test_metadata_loader.py` | MetadataLoader | 14 |
| `test_class_mapper.py` | ClassMapper | 26 |
| `test_image_mapper.py` | ImageMapper | 18 |
| `test_registry_generator.py` | RegistryGenerator | 10 |
| `test_summary_generator.py` | SummaryGenerator | 16 |

---

## Future Module Integration

Module 2 outputs are designed as stable contracts:

- **Module 3 (Preprocessing):** Load `master_metadata.csv`, iterate by `internal_id`
- **Module 4 (Augmentation):** Filter by `modality` and `normalized_class`
- **Module 5+ (Training):** Use `split` column for train/test splits
- **Module N (Inference):** Use `image_path` for loading, `internal_id` for tracking
