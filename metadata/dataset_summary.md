# Dataset Summary

> Generated: 2026-07-30 12:01 UTC

---

## Overview

| Metric | Value |
|---|---|
| Total Images | 117,917 |
| Valid Images | 117,828 |
| Invalid Images | 89 |
| Total Modalities | 7 |
| Total Normalized Classes | 17 |
| Dataset Splits | unspecified, test, train |
| Total Dataset Size | 10.42 GB |

## Images per Modality

| Modality | Total | Valid | Classes | Avg Width | Avg Height |
|---|---|---|---|---|---|
| brain_ct | 200 | 200 | 1 | 356 px | 434 px |
| brain_mri | 7,200 | 7,200 | 4 | 453 px | 457 px |
| chest_xray | 60,330 | 60,241 | 5 | 347 px | 347 px |
| histopathology | 25,000 | 25,000 | 5 | 768 px | 768 px |
| retina | 14,392 | 14,392 | 1 | 1602 px | 1235 px |
| skin | 10,015 | 10,015 | 1 | 600 px | 450 px |
| ultrasound | 780 | 780 | 3 | 616 px | 501 px |

## Images per Disease Class

| Normalized Class | Count | Modalities |
|---|---|---|
| unknown | 42,407 | chest_xray, retina, skin |
| normal | 20,517 | chest_xray, ultrasound |
| lung_opacity | 12,024 | chest_xray |
| covid | 7,232 | chest_xray |
| lung_squamous_cell | 5,000 | histopathology |
| lung_normal | 5,000 | histopathology |
| lung_adenocarcinoma | 5,000 | histopathology |
| colon_normal | 5,000 | histopathology |
| colon_adenocarcinoma | 5,000 | histopathology |
| viral_pneumonia | 2,690 | chest_xray |
| notumor | 1,800 | brain_mri |
| pituitary | 1,800 | brain_mri |
| glioma | 1,800 | brain_mri |
| meningioma | 1,800 | brain_mri |
| benign | 437 | ultrasound |
| malignant | 210 | ultrasound |
| head_ct | 200 | brain_ct |

## Dataset Sources

| Dataset Name | Images | Modality |
|---|---|---|
| COVID-19 Radiography Database | 42,330 | chest_xray |
| Lung and Colon Cancer Histopathological Images (LC25000) | 25,000 | histopathology |
| VinBigData Chest X-ray Abnormalities Detection | 18,000 | chest_xray |
| Human Against Machine with 10000 training images (HAM10000) | 10,015 | skin |
| Ocular Disease Intelligent Recognition (ODIR-5K) | 8,000 | retina |
| Retina Fundus Images (Preprocessed) | 6,392 | retina |
| glioma | 1,800 | brain_mri |
| meningioma | 1,800 | brain_mri |
| notumor | 1,800 | brain_mri |
| pituitary | 1,800 | brain_mri |
| Breast Ultrasound Images Dataset (BUSI) with Ground Truth | 780 | ultrasound |
| Head CT Scan Dataset | 200 | brain_ct |

**Largest dataset:** COVID-19 Radiography Database (42,330 images)
**Smallest dataset:** Head CT Scan Dataset (200 images)

## Image Resolution Statistics

| Metric | Width (px) | Height (px) |
|---|---|---|
| Min | 134 | 135 |
| Max | 5184 | 3456 |
| Mean | 619.4 | 561.5 |
| Median | 512 | 512 |

## Supported File Extensions

| Extension | Count |
|---|---|
| .png | 61,310 |
| .jpg | 31,607 |
| .jpeg | 25,000 |

## Data Quality

| Metric | Value |
|---|---|
| Valid Images | 117,828 (99.9%) |
| Invalid / Corrupt Images | 89 (0.1%) |

---

## Usage by Downstream Modules

All modules (3–24) should load `metadata/master_metadata.csv` as their
primary data source.  The `internal_id` column provides stable cross-module
references.  Original file paths are preserved in `image_path`.

```python
import pandas as pd
master = pd.read_csv('metadata/master_metadata.csv')
brain_mri = master[master['modality'] == 'brain_mri']
```
