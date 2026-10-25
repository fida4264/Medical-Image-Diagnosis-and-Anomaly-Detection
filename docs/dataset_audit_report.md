# Dataset Audit Report
## Medical Image Diagnosis and Anomaly Detection Platform
### Audit Date: 2026-08-19 | Read-Only Inspection

---

## Executive Summary

A complete read-only audit was performed on all seven medical imaging modality datasets
present under `datasets/`. Total: **117,917 image files** across **7 modalities**.
Usable images for supervised learning: approx **68,752** after excluding masks and unlabelled sets.

**Key findings:**

- **2 modalities are immediately READY** (Brain MRI, Histopathology)
- **4 modalities are READY WITH CONDITIONS** (Brain CT, Chest X-ray, Skin, Ultrasound)
- **1 modality is NOT READY** (Retina)
- **CRITICAL-1**: 21,165 COVID mask images contaminating master_metadata.csv as training images
- **CRITICAL-2**: 10,015 skin images have disease_class=unknown (HAM10000 labels never joined)
- **0 corrupt images. 0 zero-byte files.**

---

## Dataset Inventory

| Modality | Total Files | Net Images | Tiny/Invalid | Labelled | Unlabelled | Classes |
|----------|------------:|-----------:|-------------:|---------:|-----------:|--------:|
| Brain MRI | 7,200 | 7,200 | 0 | 7,200 | 0 | 4 |
| Brain CT | 200 | 200 | 0 | 200 | 0 | 2 |
| Chest X-ray | 60,330 | 21,165 | 89 | 21,165 | 18,000 | 4/15 |
| Histopathology | 25,000 | 25,000 | 0 | 25,000 | 0 | 5 |
| Retina | 14,392 | 14,392 | 0 | 6,392 | 8,000 | 8 |
| Skin | 10,015 | 10,015 | 0 | 10,015 | 0 | 7 |
| Ultrasound | 1,578 | 780 | 0 | 780 | 0 | 3 |
| **TOTAL** | **117,917** | **78,752** | **89** | **70,752** | **26,000** | - |

Notes:
- Chest X-ray 21,165 = COVID-19 classification images only; 18,000 VinBigData = detection format (unlabelled for classification)
- Retina: multi-label; 6,392 from full_df.csv only; 8,000 ODIR images are multi-label or competition test set
- Skin: labels verified but NOT joined to image paths in master_metadata.csv

---

## Modality-by-Modality Analysis

---

### 1. Brain MRI

**Path:** `datasets/brain_mri/`
**Dataset:** Brain Tumor MRI Dataset (Kaggle)
**Label Method:** Folder-based

| Split | glioma | meningioma | notumor | pituitary | Total |
|-------|-------:|-----------:|--------:|----------:|------:|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |
| Total | 1,800 | 1,800 | 1,800 | 1,800 | 7,200 |

**Label Status: A - VERIFIED LABELLED DATA**
Labels in folder names. 4 classes. Perfectly balanced. Pre-split (Training/ and Testing/).

- Format: `.jpg`, avg 453x457 px
- Corrupt: 0 | Zero-byte: 0 | Mask files: 0
- Patient-level leakage: NOT VERIFIED (no patient IDs available)
- Imbalance ratio: 1.0:1

**Training Readiness: READY**

---

### 2. Brain CT

**Path:** `datasets/brain_ct/head_ct/`
**Dataset:** Head CT Hemorrhage Dataset (Kaggle)
**Label Method:** External CSV - `datasets/brain_ct/labels.csv`

| Class | Count |
|-------|------:|
| hemorrhage=1 | 100 |
| hemorrhage=0 | 100 |
| Total | 200 |

**Label Status: A - VERIFIED LABELLED DATA**
labels.csv maps file id (0-199) to binary hemorrhage label. 100% coverage.

- Format: `.png`, avg 356x434 px
- Corrupt: 0 | Zero-byte: 0
- No pre-existing splits
- Patient-level leakage: NOT VERIFIED

WARNING: CRITICALLY SMALL DATASET - 200 images only.
Cross-validation (5-fold) is mandatory. Transfer learning required.

**Training Readiness: READY WITH CONDITIONS**

---

### 3. Chest X-ray

**Path:** `datasets/chest_xray/`

#### 3A. COVID-19 Radiography Dataset (Classification-Ready Subset)

| Class | Images |
|-------|-------:|
| Normal | 10,192 |
| Lung_Opacity | 6,012 |
| COVID | 3,616 |
| Viral_Pneumonia | 1,345 |
| Total | 21,165 |

**Label Status: A - VERIFIED LABELLED DATA** (images/ sub-dirs only)

> CRITICAL BUG: BUG-COVID-MASKS
> Each class folder has both /images/ and /masks/ sub-directories.
> The /masks/ directories contain 21,165 lung segmentation PNGs with normal filenames
> (no _mask suffix). The scanner _mask filename filter does NOT catch them.
> CONFIRMED: master_metadata.csv contains 21,165 rows where image_path includes /masks/.
> These mask images are indexed as training images. This is a BLOCKER.

Class imbalance ratio: 7.6:1 (Normal vs Viral_Pneumonia) - class weighting needed.

#### 3B. VinBigData (Detection Format Only)

| Property | Value |
|----------|-------|
| Train images | 15,000 |
| Test images | 3,000 |
| Format | Bounding box detection (67,914 annotation rows) |
| Classes | 15 pathologies + No finding |

**Label Status: B - UNLABELLED for classification**
Detection bounding box format. Cannot be used for single-label classification without a strategy decision.

- Corrupt: 0 | Tiny files: 89 (all in COVID /masks/ dir)
- Patient-level leakage: NOT VERIFIED

**Training Readiness: READY WITH CONDITIONS** (COVID subset only, after mask fix)

---

### 4. Histopathology

**Path:** `datasets/histopathology/lung_colon_image_set/`
**Dataset:** LC25000 Lung and Colon Cancer Histopathological Images
**Label Method:** Folder-based

| Class | Count |
|-------|------:|
| colon_aca | 5,000 |
| colon_n | 5,000 |
| lung_aca | 5,000 |
| lung_n | 5,000 |
| lung_scc | 5,000 |
| Total | 25,000 |

**Label Status: A - VERIFIED LABELLED DATA**
Folder-based labels. Perfectly balanced. No pre-existing splits.

- Format: `.jpeg`, avg 768x768 px
- Corrupt: 0 | Zero-byte: 0 | Mask files: 0
- Patient-level leakage: NOT VERIFIED (LC25000 derived from 250 pathology slides)
- Imbalance ratio: 1.0:1

**Training Readiness: READY - Best dataset in the platform**

---

### 5. Retina

**Path:** `datasets/retina/`

> UNLABELLED RETINAL DATA - NOT CURRENTLY SUITABLE FOR SUPERVISED DISEASE CLASSIFICATION

Sub-datasets:
- `ODIR-5K/ODIR-5K/Training Images/` - 7,000 images (multi-label annotations in full_df.csv)
- `ODIR-5K/ODIR-5K/Testing Images/` - 1,000 images (NO labels - competition test set)
- `preprocessed_images/` - 6,392 images (FULLY UNLABELLED, source unknown)

#### ODIR-5K Multi-Label Class Breakdown

| Code | Condition | Positive Images |
|------|-----------|----------------:|
| N | Normal | 2,101 |
| D | Diabetic Retinopathy | 2,123 |
| G | Glaucoma | 397 |
| C | Cataract | 402 |
| A | AMD | 319 |
| H | Hypertension | 203 |
| M | Myopia | 306 |
| O | Other | 1,588 |

**Label Status: D - UNCERTAIN / MULTI-LABEL** (ODIR-5K)
**Label Status: B - UNLABELLED** (preprocessed_images)

Problems:
1. Multi-label: each image can have multiple simultaneous conditions
2. Test set: 1,000 Testing Images have no labels in full_df.csv
3. Flat folder structure: scanner assigns all images disease_class=unknown
4. 6,392 images fully unlabelled (preprocessed_images/ - unknown source/license)

**Patient-Level Leakage: CONFIRMED RISK**
full_df.csv contains 3,358 unique patient IDs for 6,392 images.
Each patient has left AND right eye images. Any split must be patient-stratified.

Possible future uses (DO NOT IMPLEMENT NOW):
- Multi-label classification pipeline
- Anomaly detection (unsupervised on Normal-labelled images)
- Self-supervised representation learning (SimCLR, BYOL, MAE)
- Human-assisted annotation campaign

**Training Readiness: NOT READY**

---

### 6. Skin

**Path:** `datasets/skin/`
**Dataset:** HAM10000 Human Against Machine with 10000 training images
**Label Method:** External CSV - `HAM10000_metadata.csv` (NOT folder-based)

| dx Code | Diagnosis | Count | Confirmed By |
|---------|-----------|------:|--------------|
| nv | Melanocytic nevi | 6,705 | histo+follow_up |
| mel | Melanoma | 1,113 | histo+follow_up |
| bkl | Benign keratosis | 1,099 | histo+follow_up |
| bcc | Basal cell carcinoma | 514 | histo+consensus |
| akiec | Actinic keratoses | 327 | histo+consensus |
| vasc | Vascular lesions | 142 | histo |
| df | Dermatofibroma | 115 | histo |
| Total | | 10,015 | |

Dx type: histo=5340 (53%), follow_up=3704 (37%), consensus=902 (9%), confocal=69 (1%)

**Label Status: A - VERIFIED LABELLED DATA** (via HAM10000_metadata.csv)
All 10,015 images have clinically verified labels. All confirmation methods are valid.

WARNING: Labels NOT joined to image paths.
Images are in flat directories (HAM10000_images_part_1/, HAM10000_images_part_2/).
Scanner assigns all images disease_class=unknown.
skin_label_map.json confirms: {"0": "unknown"}.
The label join step on image_id from HAM10000_metadata.csv has never been executed.

Imbalance ratio: 58.3:1 (nv=6705 vs df=115) - severe imbalance. Class weighting mandatory.

**Patient-Level Leakage: CONFIRMED RISK**
7,470 unique lesion_ids for 10,015 images.
2,545 images are additional views of the same lesions.
Splitting must be at lesion_id level, not image_id level.

**Training Readiness: READY WITH CONDITIONS**

---

### 7. Ultrasound

**Path:** `datasets/Ultrasound/Dataset_BUSI_with_GT/`
**Dataset:** BUSI Breast Ultrasound Images Dataset with Ground Truth
**Label Method:** Folder-based

| Class | Images (net) | Mask files |
|-------|-------------:|-----------:|
| benign | 437 | 454 |
| malignant | 210 | 211 |
| normal | 133 | 133 |
| Total | 780 | 798 |

Note: benign has 454 masks for 437 images (some lesions have multiple GT masks).
scanner.py correctly excludes all 798 _mask files via filename filter.

**Label Status: A - VERIFIED LABELLED DATA**
Folder-based labels. No pre-existing splits.

WARNING: Folder is named 'Ultrasound' (capital U).
config.py KNOWN_MODALITIES expects 'ultrasound' (lowercase).
On Linux / Docker, this will cause the scanner to report the modality as missing.

- Corrupt: 0 | Zero-byte: 0
- Imbalance ratio: 3.3:1 - class weighting required
- Patient-level leakage: NOT VERIFIED
- Very small dataset: 780 images total

**Training Readiness: READY WITH CONDITIONS**

---

## Critical Issues

| ID | Severity | Modality | Issue |
|----|----------|----------|-------|
| CRITICAL-1 | ERROR | Chest X-ray | 21,165 COVID mask images indexed in master_metadata.csv as training images |
| CRITICAL-2 | ERROR | Skin | 10,015 skin images have disease_class=unknown - HAM10000 labels never joined |
| CRITICAL-3 | WARNING | Retina | Multi-label format + unlabelled subset - not suitable for supervised classification |
| CRITICAL-4 | WARNING | Skin, Retina | Patient-level leakage risk (lesion-level skin, patient-level retina) |
| CRITICAL-5 | WARNING | Ultrasound | Folder 'Ultrasound' (capital U) vs config 'ultrasound' - Linux/Docker failure risk |
| CRITICAL-6 | INFO | Chest X-ray | VinBigData is detection format only - no classification strategy defined |
| CRITICAL-7 | WARNING | Brain CT | 200 images - critically small for deep learning training |

---

## Training Readiness

| Modality | Verdict | Condition |
|----------|---------|-----------|
| Brain MRI | READY | None |
| Histopathology | READY | None |
| Ultrasound | READY WITH CONDITIONS | Cross-validation + class weights required |
| Brain CT | READY WITH CONDITIONS | Cross-validation required; only 200 images |
| Chest X-ray | READY WITH CONDITIONS | Fix CRITICAL-1 (mask path filter) first; COVID subset only |
| Skin | READY WITH CONDITIONS | Fix CRITICAL-2 (join HAM10000 labels); lesion-level split required |
| Retina | NOT READY | Explicit design decision required before any pipeline work |

---

## Recommended Next Action

| Modality | Next Action |
|----------|-------------|
| Brain MRI | Begin training pipeline immediately |
| Histopathology | Begin training pipeline immediately |
| Brain CT | Define cross-validation strategy |
| Ultrasound | Define cross-validation strategy; apply class weights |
| Chest X-ray | Fix CRITICAL-1 (add /masks/ path filter); use COVID subset only |
| Skin | Fix CRITICAL-2 (join HAM10000_metadata.csv labels); add lesion-level split |
| Retina | User decision required: (A) multi-label pipeline, (B) anomaly detection only, (C) annotation campaign |

---

## Existing Infrastructure Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| DatasetScanner | WORKING | Excludes _mask filenames; does NOT exclude /masks/ directories |
| DatasetValidator | WORKING | Corruption, duplicate, empty folder checks |
| LabelGenerator | WORKING | Folder-based labels only; no CSV-join capability |
| master_metadata.csv | CONTAMINATED | 21,165 COVID mask paths indexed as training images |
| modality_splits.csv | INCOMPLETE | Skin=unknown; mask rows included |
| skin label_map.json | INCORRECT | Shows 0=unknown - labels were never joined |
| retina label_map.json | CORRECT | Shows 0=unknown - accurately reflects unlabelled state |
| DatasetStatistics | WORKING | Image dimension and class distribution statistics |
| dataset_registry.json | ACCURATE | Dataset sources and licenses documented |

Missing capabilities:
1. Skin label join (HAM10000_metadata.csv -> image_path on image_id)
2. /masks/ directory path filter in scanner or metadata loader
3. Multi-label classification pipeline for retina
4. Lesion-level and patient-level stratified splitting

---

## Test Results

| Test File | Passed | Failed |
|-----------|-------:|-------:|
| test_scanner.py | 23 | 0 |
| test_validator.py | 12 | 0 |
| test_label_generator.py | 14 | 0 |
| test_statistics.py | 18 | 0 |
| test_metadata_generator.py | 14 | 0 |
| Total | 81 | 0 |

No regressions. All 81 dataset management tests pass after the audit.

---

Audit generated: 2026-08-19
Read-only inspection - no datasets modified, moved, or deleted.
Files created: docs/dataset_audit_report.md, outputs/dataset_audit_summary.json
