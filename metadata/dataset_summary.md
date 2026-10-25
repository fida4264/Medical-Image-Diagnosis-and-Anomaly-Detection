# Dataset Summary

> Generated: 2026-08-19 05:39 UTC

---

## Overview

| Metric | Value |
|---|---|
| Total Images | 96,752 |
| Valid Images | 96,752 |
| Invalid Images | 0 |
| Total Modalities | 7 |
| Total Normalized Classes | 24 |
| Dataset Splits | unspecified, test, train |
| Total Dataset Size | 10.39 GB |

## Images per Modality

| Modality | Total | Valid | Classes | Avg Width | Avg Height |
|---|---|---|---|---|---|
| brain_ct | 200 | 200 | 1 | 356 px | 434 px |
| brain_mri | 7,200 | 7,200 | 4 | 453 px | 457 px |
| chest_xray | 39,165 | 39,165 | 5 | 397 px | 397 px |
| histopathology | 25,000 | 25,000 | 5 | 768 px | 768 px |
| retina | 14,392 | 14,392 | 1 | 1602 px | 1235 px |
| skin | 10,015 | 10,015 | 7 | 600 px | 450 px |
| ultrasound | 780 | 780 | 3 | 616 px | 501 px |

## Images per Disease Class

| Normalized Class | Count | Modalities |
|---|---|---|
| unknown | 32,392 | chest_xray, retina |
| normal | 10,325 | chest_xray, ultrasound |
| melanocytic_nevus | 6,705 | skin |
| lung_opacity | 6,012 | chest_xray |
| colon_normal | 5,000 | histopathology |
| colon_adenocarcinoma | 5,000 | histopathology |
| lung_adenocarcinoma | 5,000 | histopathology |
| lung_squamous_cell | 5,000 | histopathology |
| lung_normal | 5,000 | histopathology |
| covid | 3,616 | chest_xray |
| pituitary | 1,800 | brain_mri |
| glioma | 1,800 | brain_mri |
| notumor | 1,800 | brain_mri |
| meningioma | 1,800 | brain_mri |
| viral_pneumonia | 1,345 | chest_xray |
| melanoma | 1,113 | skin |
| benign_keratosis | 1,099 | skin |
| basal_cell_carcinoma | 514 | skin |
| benign | 437 | ultrasound |
| actinic_keratosis | 327 | skin |
| malignant | 210 | ultrasound |
| head_ct | 200 | brain_ct |
| vascular_lesion | 142 | skin |
| dermatofibroma | 115 | skin |

## Dataset Sources

| Dataset Name | Images | Modality |
|---|---|---|
| Lung and Colon Cancer Histopathological Images (LC25000) | 25,000 | histopathology |
| COVID-19 Radiography Database | 21,165 | chest_xray |
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

**Largest dataset:** Lung and Colon Cancer Histopathological Images (LC25000) (25,000 images)
**Smallest dataset:** Head CT Scan Dataset (200 images)

## Image Resolution Statistics

| Metric | Width (px) | Height (px) |
|---|---|---|
| Min | 134 | 135 |
| Max | 5184 | 3456 |
| Mean | 698.9 | 628.3 |
| Median | 512 | 512 |

## Supported File Extensions

| Extension | Count |
|---|---|
| .png | 40,145 |
| .jpg | 31,607 |
| .jpeg | 25,000 |

## Data Quality

| Metric | Value |
|---|---|
| Valid Images | 96,752 (100.0%) |
| Invalid / Corrupt Images | 0 (0.0%) |

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
