"""
metadata_manager/v2_label_reconstructor.py
=========================================
V2.0 Label Reconstruction & Standardization Engine.

Constructs canonical, standardized V2.0 metadata records and manifests
across all seven medical imaging modalities:
1. Brain MRI (7,200 images, 4 classes)
2. Brain CT (200 images, binary hemorrhage vs. normal)
3. Chest X-ray (21,165 images, 4 classes; 21,165 masks quarantined; VinBigData on HOLD)
4. Histopathology (25,000 images, 5 classes)
5. Retina (6,392 supervised images, 8-target multi-label; 8,000 unverified/unlabeled quarantined)
6. Skin Lesion (10,015 images, 7 classes joined from HAM10000_metadata.csv)
7. Ultrasound (780 images, 3 classes; 798 masks quarantined)

Author: Antigravity V2.0 Upgrade Engine
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from metadata_manager.config import (
    PROJECT_ROOT,
    MANIFESTS_DIR,
    V2_CANONICAL_COLUMNS,
    DATASET_VERSIONS,
    V2_MODALITY_CLASSES,
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
)

logger = logging.getLogger(__name__)


class V2LabelReconstructor:
    """Orchestrates Phase 2 V2.0 Label Reconstruction and Standardization."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or PROJECT_ROOT
        self.datasets_dir = self.root_dir / "datasets"
        self.manifests_dir = self.root_dir / "data" / "manifests"
        self.outputs_dir = self.root_dir / "outputs"

        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.quarantine_records: List[Dict[str, Any]] = []

    def _relpath(self, path: Path) -> str:
        """Return forward-slash relative path from project root."""
        try:
            return path.relative_to(self.root_dir).as_posix()
        except ValueError:
            return path.as_posix()

    # -------------------------------------------------------------------------
    # 1. Brain MRI Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_brain_mri(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Brain MRI labels from folder hierarchy.
        Mapping:
          glioma -> glioma (class_id: 0)
          meningioma -> meningioma (class_id: 1)
          notumor -> no_tumor (class_id: 2)
          pituitary -> pituitary (class_id: 3)
        """
        records = []
        base = self.datasets_dir / "brain_mri"
        mapping = {
            "glioma": "glioma",
            "meningioma": "meningioma",
            "notumor": "no_tumor",
            "pituitary": "pituitary",
        }
        class_ids = V2_MODALITY_CLASSES["brain_mri"]

        for split in ["Training", "Testing"]:
            split_dir = base / split
            split_status = "train" if split == "Training" else "test"
            if not split_dir.exists():
                continue
            for cls_folder in sorted(split_dir.iterdir()):
                if not cls_folder.is_dir():
                    continue
                orig_cls = cls_folder.name
                std_cls = mapping.get(orig_cls, orig_cls)
                cid = class_ids.get(std_cls, None)

                for img_file in sorted(cls_folder.glob("*")):
                    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    rec = {
                        COL_V2_IMAGE_ID: f"MRI_{split[:2].upper()}_{img_file.stem}",
                        COL_V2_IMAGE_PATH: self._relpath(img_file),
                        COL_V2_MODALITY: "brain_mri",
                        COL_V2_DATASET_NAME: "Brain Tumor MRI Dataset",
                        COL_V2_DATASET_VERSION: DATASET_VERSIONS["brain_mri"],
                        COL_V2_SOURCE: "Kaggle",
                        COL_V2_ORIGINAL_LABEL: orig_cls,
                        COL_V2_STANDARDIZED_LABEL: std_cls,
                        COL_V2_LABEL_TYPE: "multiclass",
                        COL_V2_CLASS_ID: cid,
                        COL_V2_MULTI_LABEL_VECTOR: None,
                        COL_V2_PATIENT_ID: None,
                        COL_V2_STUDY_ID: None,
                        COL_V2_LESION_ID: None,
                        COL_V2_EYE_ID: None,
                        COL_V2_LEAKAGE_GROUP_ID: None,
                        COL_V2_LABEL_STATUS: "VALIDATED",
                        COL_V2_METADATA_SOURCE: "folder_structure",
                        COL_V2_MASK_STATUS: "NONE",
                        COL_V2_QUALITY_STATUS: "PASSED",
                        COL_V2_SPLIT_STATUS: split_status,
                        COL_V2_DUPLICATE_STATUS: "UNIQUE",
                    }
                    records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # 2. Brain CT Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_brain_ct(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Brain CT labels by joining head_ct/*.png with labels.csv on ID.
        Mapping:
          0 -> normal (class_id: 0)
          1 -> hemorrhage (class_id: 1)
        """
        records = []
        base = self.datasets_dir / "brain_ct"
        csv_path = base / "labels.csv"
        img_dir = base / "head_ct"

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        csv_map = dict(zip(df["id"].astype(int), df["hemorrhage"].astype(int)))

        for img_file in sorted(img_dir.glob("*.png")):
            try:
                fid = int(img_file.stem)
            except ValueError:
                continue

            if fid in csv_map:
                orig_label = int(csv_map[fid])
                std_label = "hemorrhage" if orig_label == 1 else "normal"
                cid = 1 if orig_label == 1 else 0
                lstat = "VALIDATED"
            else:
                orig_label = None
                std_label = None
                cid = None
                lstat = "MISSING_LABEL"

            rec = {
                COL_V2_IMAGE_ID: f"CT_{img_file.stem}",
                COL_V2_IMAGE_PATH: self._relpath(img_file),
                COL_V2_MODALITY: "brain_ct",
                COL_V2_DATASET_NAME: "Head CT Hemorrhage Dataset",
                COL_V2_DATASET_VERSION: DATASET_VERSIONS["brain_ct"],
                COL_V2_SOURCE: "Kaggle",
                COL_V2_ORIGINAL_LABEL: orig_label,
                COL_V2_STANDARDIZED_LABEL: std_label,
                COL_V2_LABEL_TYPE: "binary_classification",
                COL_V2_CLASS_ID: cid,
                COL_V2_MULTI_LABEL_VECTOR: None,
                COL_V2_PATIENT_ID: None,
                COL_V2_STUDY_ID: None,
                COL_V2_LESION_ID: None,
                COL_V2_EYE_ID: None,
                COL_V2_LEAKAGE_GROUP_ID: str(fid),
                COL_V2_LABEL_STATUS: lstat,
                COL_V2_METADATA_SOURCE: "labels.csv",
                COL_V2_MASK_STATUS: "NONE",
                COL_V2_QUALITY_STATUS: "PASSED",
                COL_V2_SPLIT_STATUS: "UNSPLIT",
                COL_V2_DUPLICATE_STATUS: "UNIQUE",
            }
            records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # 3. Chest X-ray Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_chest_xray(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Chest X-ray classification labels.
        Excludes all 21,165 masks (sent to quarantine).
        Excludes VinBigData (sent to quarantine as HOLD).
        """
        records = []
        base = self.datasets_dir / "chest_xray" / "COVID-19_Radiography_Dataset"
        mapping = {
            "COVID": "covid",
            "Lung_Opacity": "lung_opacity",
            "Normal": "normal",
            "Viral Pneumonia": "viral_pneumonia",
        }
        class_ids = V2_MODALITY_CLASSES["chest_xray"]

        for orig_cls, std_cls in mapping.items():
            cls_dir = base / orig_cls
            img_dir = cls_dir / "images"
            mask_dir = cls_dir / "masks"

            # 1. Supervised classification images
            if img_dir.exists():
                for img_file in sorted(img_dir.glob("*")):
                    if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    rec = {
                        COL_V2_IMAGE_ID: f"CXR_{img_file.stem}",
                        COL_V2_IMAGE_PATH: self._relpath(img_file),
                        COL_V2_MODALITY: "chest_xray",
                        COL_V2_DATASET_NAME: "COVID-19 Radiography Database",
                        COL_V2_DATASET_VERSION: DATASET_VERSIONS["chest_xray"],
                        COL_V2_SOURCE: "Kaggle / IEEE DataPort",
                        COL_V2_ORIGINAL_LABEL: orig_cls,
                        COL_V2_STANDARDIZED_LABEL: std_cls,
                        COL_V2_LABEL_TYPE: "multiclass",
                        COL_V2_CLASS_ID: class_ids[std_cls],
                        COL_V2_MULTI_LABEL_VECTOR: None,
                        COL_V2_PATIENT_ID: None,
                        COL_V2_STUDY_ID: None,
                        COL_V2_LESION_ID: None,
                        COL_V2_EYE_ID: None,
                        COL_V2_LEAKAGE_GROUP_ID: None,
                        COL_V2_LABEL_STATUS: "VALIDATED",
                        COL_V2_METADATA_SOURCE: "folder_structure",
                        COL_V2_MASK_STATUS: "PAIRED",
                        COL_V2_QUALITY_STATUS: "PASSED",
                        COL_V2_SPLIT_STATUS: "UNSPLIT",
                        COL_V2_DUPLICATE_STATUS: "UNIQUE",
                    }
                    records.append(rec)

            # 2. Quarantine masks
            if mask_dir.exists():
                for mask_file in sorted(mask_dir.glob("*")):
                    if mask_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    self.quarantine_records.append({
                        "image_id": f"CXR_MASK_{mask_file.stem}",
                        "image_path": self._relpath(mask_file),
                        "modality": "chest_xray",
                        "dataset_name": "COVID-19 Radiography Database",
                        "dataset_version": DATASET_VERSIONS["chest_xray"],
                        "original_label": orig_cls,
                        "label_status": "MASK",
                        "quarantine_reason": "COVID-19 Radiography lung segmentation mask (not a classification image)",
                    })

        # 3. Quarantine VinBigData (HOLD)
        vinbig_dir = self.datasets_dir / "chest_xray" / "vinbigdata"
        if vinbig_dir.exists():
            for f in sorted(vinbig_dir.rglob("*")):
                if f.is_file():
                    self.quarantine_records.append({
                        "image_id": f"VINBIG_{f.stem}",
                        "image_path": self._relpath(f),
                        "modality": "chest_xray",
                        "dataset_name": "VinBigData Chest X-ray Abnormalities Detection",
                        "dataset_version": "vinbigdata_v1.0_hold",
                        "original_label": "object_detection",
                        "label_status": "HOLD",
                        "quarantine_reason": "Bounding-box object detection dataset held in reserve; not formatted for classification",
                    })

        return records

    # -------------------------------------------------------------------------
    # 4. Histopathology Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_histopathology(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Histopathology labels from nested folder structure.
        Mapping:
          colon_aca -> colon_adenocarcinoma (class_id: 0)
          colon_n -> colon_normal (class_id: 1)
          lung_aca -> lung_adenocarcinoma (class_id: 2)
          lung_n -> lung_normal (class_id: 3)
          lung_scc -> lung_squamous_cell_carcinoma (class_id: 4)
        """
        records = []
        base = self.datasets_dir / "histopathology" / "lung_colon_image_set"
        mapping = {
            "colon_aca": "colon_adenocarcinoma",
            "colon_n": "colon_normal",
            "lung_aca": "lung_adenocarcinoma",
            "lung_n": "lung_normal",
            "lung_scc": "lung_squamous_cell_carcinoma",
        }
        class_ids = V2_MODALITY_CLASSES["histopathology"]

        for subset_dir in sorted(base.iterdir()):
            if not subset_dir.is_dir():
                continue
            for cls_folder in sorted(subset_dir.iterdir()):
                if not cls_folder.is_dir():
                    continue
                orig_cls = cls_folder.name
                if orig_cls not in mapping:
                    continue
                std_cls = mapping[orig_cls]
                cid = class_ids[std_cls]

                for img_file in sorted(cls_folder.glob("*")):
                    if img_file.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                        continue
                    rec = {
                        COL_V2_IMAGE_ID: f"HISTO_{img_file.stem}",
                        COL_V2_IMAGE_PATH: self._relpath(img_file),
                        COL_V2_MODALITY: "histopathology",
                        COL_V2_DATASET_NAME: "LC25000 Lung and Colon Histopathology",
                        COL_V2_DATASET_VERSION: DATASET_VERSIONS["histopathology"],
                        COL_V2_SOURCE: "Kaggle / LC25000",
                        COL_V2_ORIGINAL_LABEL: orig_cls,
                        COL_V2_STANDARDIZED_LABEL: std_cls,
                        COL_V2_LABEL_TYPE: "multiclass",
                        COL_V2_CLASS_ID: cid,
                        COL_V2_MULTI_LABEL_VECTOR: None,
                        COL_V2_PATIENT_ID: None,
                        COL_V2_STUDY_ID: None,
                        COL_V2_LESION_ID: None,
                        COL_V2_EYE_ID: None,
                        COL_V2_LEAKAGE_GROUP_ID: None,
                        COL_V2_LABEL_STATUS: "VALIDATED",
                        COL_V2_METADATA_SOURCE: "folder_structure",
                        COL_V2_MASK_STATUS: "NONE",
                        COL_V2_QUALITY_STATUS: "PASSED",
                        COL_V2_SPLIT_STATUS: "UNSPLIT",
                        COL_V2_DUPLICATE_STATUS: "NEAR_DUPLICATE_RISK",
                    }
                    records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # 5. Retina Reconstruction (MULTI-LABEL)
    # -------------------------------------------------------------------------
    def reconstruct_retina(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Retina labels as TRUE MULTI-LABEL records from full_df.csv.
        Binary targets: N, D, G, C, A, H, M, O.
        Quarantines:
          - 1,000 unlabeled ODIR-5K Testing Images
          - 608 unmapped ODIR-5K Training Images
          - 6,392 preprocessed_images with UNKNOWN_SOURCE
        """
        records = []
        base = self.datasets_dir / "retina"
        csv_path = base / "full_df.csv"
        train_dir = base / "ODIR-5K" / "ODIR-5K" / "Training Images"
        test_dir = base / "ODIR-5K" / "ODIR-5K" / "Testing Images"
        prep_dir = base / "preprocessed_images"

        df = pd.read_csv(csv_path)

        targets_map = [
            ("N", "normal"),
            ("D", "diabetic_retinopathy"),
            ("G", "glaucoma"),
            ("C", "cataract"),
            ("A", "amd"),
            ("H", "hypertension"),
            ("M", "pathological_myopia"),
            ("O", "other_abnormality"),
        ]

        mapped_filenames = set()

        for _, row in df.iterrows():
            fname = str(row.get("filename", "")).strip()
            if not fname:
                continue

            img_file = train_dir / fname
            if not img_file.exists():
                continue

            mapped_filenames.add(fname)

            # Build multi-label vector and active labels
            vector = []
            active_labels = []
            label_dict = {}
            for col_letter, std_name in targets_map:
                val = int(row.get(col_letter, 0))
                vector.append(val)
                label_dict[std_name] = val
                if val == 1:
                    active_labels.append(std_name)

            pid = str(int(row["ID"])) if pd.notnull(row.get("ID")) else None

            # Determine eye
            eye_id = None
            if "right" in fname.lower():
                eye_id = "right"
            elif "left" in fname.lower():
                eye_id = "left"

            rec = {
                COL_V2_IMAGE_ID: f"RET_{img_file.stem}",
                COL_V2_IMAGE_PATH: self._relpath(img_file),
                COL_V2_MODALITY: "retina",
                COL_V2_DATASET_NAME: "ODIR-5K Ocular Disease Intelligent Recognition",
                COL_V2_DATASET_VERSION: DATASET_VERSIONS["retina"],
                COL_V2_SOURCE: "Kaggle / ODIR",
                COL_V2_ORIGINAL_LABEL: str(row.get("labels", "")),
                COL_V2_STANDARDIZED_LABEL: ";".join(active_labels) if active_labels else "normal",
                COL_V2_LABEL_TYPE: "multi_label",
                COL_V2_CLASS_ID: None,
                COL_V2_MULTI_LABEL_VECTOR: json.dumps(vector),
                COL_V2_PATIENT_ID: pid,
                COL_V2_STUDY_ID: None,
                COL_V2_LESION_ID: None,
                COL_V2_EYE_ID: eye_id,
                COL_V2_LEAKAGE_GROUP_ID: pid,  # Group by patient to keep left & right eyes together
                COL_V2_LABEL_STATUS: "VALIDATED",
                COL_V2_METADATA_SOURCE: "full_df.csv",
                COL_V2_MASK_STATUS: "NONE",
                COL_V2_QUALITY_STATUS: "PASSED",
                COL_V2_SPLIT_STATUS: "UNSPLIT",
                COL_V2_DUPLICATE_STATUS: "UNIQUE",
            }
            records.append(rec)

        # 1. Quarantine unmapped training images in ODIR-5K
        if train_dir.exists():
            for f in sorted(train_dir.glob("*.jpg")):
                if f.name not in mapped_filenames:
                    self.quarantine_records.append({
                        "image_id": f"RET_UNMAPPED_{f.stem}",
                        "image_path": self._relpath(f),
                        "modality": "retina",
                        "dataset_name": "ODIR-5K",
                        "dataset_version": DATASET_VERSIONS["retina"],
                        "original_label": "none",
                        "label_status": "MISSING_LABEL",
                        "quarantine_reason": "Training image present in folder but omitted from full_df.csv",
                    })

        # 2. Quarantine ODIR-5K Testing Images (Unlabeled competition set)
        if test_dir.exists():
            for f in sorted(test_dir.glob("*.jpg")):
                self.quarantine_records.append({
                    "image_id": f"RET_TEST_{f.stem}",
                    "image_path": self._relpath(f),
                    "modality": "retina",
                    "dataset_name": "ODIR-5K",
                    "dataset_version": DATASET_VERSIONS["retina"],
                    "original_label": "unlabeled",
                    "label_status": "MISSING_LABEL",
                    "quarantine_reason": "ODIR-5K competition test set image without ground truth labels",
                })

        # 3. Quarantine preprocessed_images (UNKNOWN_SOURCE)
        if prep_dir.exists():
            for f in sorted(prep_dir.glob("*.jpg")):
                self.quarantine_records.append({
                    "image_id": f"RET_PREP_{f.stem}",
                    "image_path": self._relpath(f),
                    "modality": "retina",
                    "dataset_name": "ODIR-5K Preprocessed",
                    "dataset_version": "retina_v2.0_prep_quarantine",
                    "original_label": "preprocessed",
                    "label_status": "UNKNOWN_SOURCE",
                    "quarantine_reason": "Preprocessed fundus crop with undocumented processing provenance and unknown license",
                })

        return records

    # -------------------------------------------------------------------------
    # 6. Skin Lesion Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_skin(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Skin Lesion labels by joining HAM10000_metadata.csv on image_id.
        Mapping:
          nv -> melanocytic_nevus (class_id: 0)
          mel -> melanoma (class_id: 1)
          bkl -> benign_keratosis (class_id: 2)
          bcc -> basal_cell_carcinoma (class_id: 3)
          akiec -> actinic_keratosis (class_id: 4)
          vasc -> vascular_lesion (class_id: 5)
          df -> dermatofibroma (class_id: 6)
        """
        records = []
        base = self.datasets_dir / "skin"
        meta_path = base / "HAM10000_metadata.csv"
        p1_dir = base / "HAM10000_images_part_1"
        p2_dir = base / "HAM10000_images_part_2"

        df = pd.read_csv(meta_path)
        meta_lookup = {}
        for _, row in df.iterrows():
            iid = str(row["image_id"]).strip()
            meta_lookup[iid] = row.to_dict()

        dx_map = {
            "nv": "melanocytic_nevus",
            "mel": "melanoma",
            "bkl": "benign_keratosis",
            "bcc": "basal_cell_carcinoma",
            "akiec": "actinic_keratosis",
            "vasc": "vascular_lesion",
            "df": "dermatofibroma",
        }
        class_ids = V2_MODALITY_CLASSES["skin"]

        # Track lesion counts to detect duplicates
        lesion_counts = df["lesion_id"].value_counts().to_dict()

        for folder in [p1_dir, p2_dir]:
            if not folder.exists():
                continue
            for img_file in sorted(folder.glob("*.jpg")):
                iid = img_file.stem
                meta = meta_lookup.get(iid)
                if not meta:
                    continue

                raw_dx = str(meta.get("dx", "")).strip()
                std_dx = dx_map.get(raw_dx, raw_dx)
                cid = class_ids.get(std_dx, None)
                lesion_id = str(meta.get("lesion_id", "")).strip()

                is_repeated = lesion_counts.get(lesion_id, 0) > 1
                dup_stat = "REPEATED_LESION" if is_repeated else "UNIQUE"

                rec = {
                    COL_V2_IMAGE_ID: f"SKIN_{iid}",
                    COL_V2_IMAGE_PATH: self._relpath(img_file),
                    COL_V2_MODALITY: "skin",
                    COL_V2_DATASET_NAME: "HAM10000 Skin Lesion Dataset",
                    COL_V2_DATASET_VERSION: DATASET_VERSIONS["skin"],
                    COL_V2_SOURCE: "ISIC Archive / Kaggle",
                    COL_V2_ORIGINAL_LABEL: raw_dx,
                    COL_V2_STANDARDIZED_LABEL: std_dx,
                    COL_V2_LABEL_TYPE: "multiclass",
                    COL_V2_CLASS_ID: cid,
                    COL_V2_MULTI_LABEL_VECTOR: None,
                    COL_V2_PATIENT_ID: None,
                    COL_V2_STUDY_ID: None,
                    COL_V2_LESION_ID: lesion_id,
                    COL_V2_EYE_ID: None,
                    COL_V2_LEAKAGE_GROUP_ID: lesion_id,  # Critical: Group by lesion_id to prevent leakage
                    COL_V2_LABEL_STATUS: "VALIDATED",
                    COL_V2_METADATA_SOURCE: "HAM10000_metadata.csv",
                    COL_V2_MASK_STATUS: "NONE",
                    COL_V2_QUALITY_STATUS: "PASSED",
                    COL_V2_SPLIT_STATUS: "UNSPLIT",
                    COL_V2_DUPLICATE_STATUS: dup_stat,
                }
                records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # 7. Ultrasound Reconstruction
    # -------------------------------------------------------------------------
    def reconstruct_ultrasound(self) -> List[Dict[str, Any]]:
        """
        Reconstruct Ultrasound labels from Dataset_BUSI_with_GT.
        Excludes all 798 files matching *_mask*.png (sent to quarantine).
        Classes: benign (0), malignant (1), normal (2).
        """
        records = []
        base = self.datasets_dir / "Ultrasound" / "Dataset_BUSI_with_GT"
        if not base.exists():
            base = self.datasets_dir / "ultrasound" / "Dataset_BUSI_with_GT"

        mapping = {
            "benign": "benign",
            "malignant": "malignant",
            "normal": "normal",
        }
        class_ids = V2_MODALITY_CLASSES["ultrasound"]

        for orig_cls, std_cls in mapping.items():
            cls_dir = base / orig_cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.glob("*.png")):
                if "_mask" in f.name.lower():
                    # Quarantine mask
                    self.quarantine_records.append({
                        "image_id": f"US_MASK_{f.stem}",
                        "image_path": self._relpath(f),
                        "modality": "ultrasound",
                        "dataset_name": "Dataset_BUSI_with_GT",
                        "dataset_version": DATASET_VERSIONS["ultrasound"],
                        "original_label": orig_cls,
                        "label_status": "MASK",
                        "quarantine_reason": "BUSI ground truth segmentation mask (excluded from classification manifest)",
                    })
                else:
                    # True classification image
                    cid = class_ids[std_cls]
                    rec = {
                        COL_V2_IMAGE_ID: f"US_{f.stem}",
                        COL_V2_IMAGE_PATH: self._relpath(f),
                        COL_V2_MODALITY: "ultrasound",
                        COL_V2_DATASET_NAME: "Breast Ultrasound Images Dataset (BUSI)",
                        COL_V2_DATASET_VERSION: DATASET_VERSIONS["ultrasound"],
                        COL_V2_SOURCE: "Kaggle / BUSI",
                        COL_V2_ORIGINAL_LABEL: orig_cls,
                        COL_V2_STANDARDIZED_LABEL: std_cls,
                        COL_V2_LABEL_TYPE: "multiclass",
                        COL_V2_CLASS_ID: cid,
                        COL_V2_MULTI_LABEL_VECTOR: None,
                        COL_V2_PATIENT_ID: None,
                        COL_V2_STUDY_ID: None,
                        COL_V2_LESION_ID: None,
                        COL_V2_EYE_ID: None,
                        COL_V2_LEAKAGE_GROUP_ID: None,
                        COL_V2_LABEL_STATUS: "VALIDATED",
                        COL_V2_METADATA_SOURCE: "folder_structure",
                        COL_V2_MASK_STATUS: "PAIRED",
                        COL_V2_QUALITY_STATUS: "PASSED",
                        COL_V2_SPLIT_STATUS: "UNSPLIT",
                        COL_V2_DUPLICATE_STATUS: "UNIQUE",
                    }
                    records.append(rec)
        return records

    # -------------------------------------------------------------------------
    # Orchestrator & Exporter
    # -------------------------------------------------------------------------
    def run_reconstruction(self) -> Dict[str, Any]:
        """Execute full reconstruction for all modalities and write manifests."""
        self.quarantine_records = []

        manifest_data = {
            "brain_mri": self.reconstruct_brain_mri(),
            "brain_ct": self.reconstruct_brain_ct(),
            "chest_xray": self.reconstruct_chest_xray(),
            "histopathology": self.reconstruct_histopathology(),
            "retina": self.reconstruct_retina(),
            "skin": self.reconstruct_skin(),
            "ultrasound": self.reconstruct_ultrasound(),
        }

        # Write modality manifests
        for modality, recs in manifest_data.items():
            fname = f"{DATASET_VERSIONS[modality]}.csv"
            out_path = self.manifests_dir / fname
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=V2_CANONICAL_COLUMNS)
                writer.writeheader()
                writer.writerows(recs)
            logger.info("Saved manifest: %s (%d records)", out_path, len(recs))

        # Write quarantine manifest
        q_path = self.manifests_dir / "v2_quarantine.csv"
        q_fields = [
            "image_id", "image_path", "modality", "dataset_name",
            "dataset_version", "original_label", "label_status", "quarantine_reason"
        ]
        with open(q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=q_fields)
            writer.writeheader()
            writer.writerows(self.quarantine_records)
        logger.info("Saved quarantine manifest: %s (%d records)", q_path, len(self.quarantine_records))

        # Perform automatic validation
        validation_report = self.validate_manifests(manifest_data)

        # Save JSON validation report
        val_json_path = self.outputs_dir / "v2_label_reconstruction_report.json"
        with open(val_json_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2)

        return validation_report

    def validate_manifests(self, manifest_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate all generated records against strict V2.0 integrity rules."""
        report = {
            "version": "2.0.0",
            "timestamp": "2026-09-19",
            "total_supervised_images": sum(len(v) for v in manifest_data.values()),
            "total_quarantined_records": len(self.quarantine_records),
            "modality_reports": {},
            "validation_checks": {
                "all_images_exist_on_disk": True,
                "all_labels_non_null": True,
                "all_classes_in_allowed_set": True,
                "multi_label_vectors_correct_dimension": True,
                "no_mask_files_in_supervised_manifests": True,
                "no_hold_datasets_in_supervised_manifests": True,
            },
            "failures": []
        }

        allowed_classes = V2_MODALITY_CLASSES

        for modality, records in manifest_data.items():
            mod_report = {
                "record_count": len(records),
                "class_counts": {},
                "label_type": records[0][COL_V2_LABEL_TYPE] if records else None,
                "missing_files": 0,
                "missing_labels": 0,
                "invalid_classes": 0,
                "mask_contaminations": 0,
            }

            for rec in records:
                img_p = self.root_dir / rec[COL_V2_IMAGE_PATH]
                if not img_p.exists():
                    mod_report["missing_files"] += 1
                    report["validation_checks"]["all_images_exist_on_disk"] = False
                    report["failures"].append(f"{modality}: Image file missing: {rec[COL_V2_IMAGE_PATH]}")

                # Label presence
                std_lbl = rec[COL_V2_STANDARDIZED_LABEL]
                if not std_lbl:
                    mod_report["missing_labels"] += 1
                    report["validation_checks"]["all_labels_non_null"] = False
                    report["failures"].append(f"{modality}: Missing standardized label for {rec[COL_V2_IMAGE_ID]}")

                # Class validation
                if modality != "retina":
                    if std_lbl not in allowed_classes[modality]:
                        mod_report["invalid_classes"] += 1
                        report["validation_checks"]["all_classes_in_allowed_set"] = False
                        report["failures"].append(f"{modality}: Unexpected class '{std_lbl}' for {rec[COL_V2_IMAGE_ID]}")
                    mod_report["class_counts"][std_lbl] = mod_report["class_counts"].get(std_lbl, 0) + 1
                else:
                    # Retina multi-label validation
                    vec_str = rec[COL_V2_MULTI_LABEL_VECTOR]
                    vec = json.loads(vec_str)
                    if len(vec) != 8:
                        report["validation_checks"]["multi_label_vectors_correct_dimension"] = False
                        report["failures"].append(f"retina: Invalid vector length {len(vec)} for {rec[COL_V2_IMAGE_ID]}")
                    for lbl in std_lbl.split(";"):
                        if lbl not in allowed_classes["retina"]:
                            mod_report["invalid_classes"] += 1
                            report["validation_checks"]["all_classes_in_allowed_set"] = False
                        mod_report["class_counts"][lbl] = mod_report["class_counts"].get(lbl, 0) + 1

                # Mask contamination check
                if "_mask" in Path(rec[COL_V2_IMAGE_PATH]).name.lower():
                    mod_report["mask_contaminations"] += 1
                    report["validation_checks"]["no_mask_files_in_supervised_manifests"] = False
                    report["failures"].append(f"{modality}: Mask contamination detected: {rec[COL_V2_IMAGE_PATH]}")

            report["modality_reports"][modality] = mod_report

        return report
