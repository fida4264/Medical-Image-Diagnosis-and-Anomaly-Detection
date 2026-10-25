"""
tests/test_v2_label_reconstruction.py
======================================
Focused unit and integration tests for V2.0 Label Reconstruction & Standardization:
- Canonical metadata schema compliance
- Brain CT CSV-to-image mapping
- Brain MRI folder mapping
- Chest X-ray class mapping (4 classes, no 'unknown')
- Histopathology nested folder mapping (5 classes)
- Retina multi-label mapping (8 binary targets)
- Retina unlabeled & unverified exclusion
- Skin metadata join (100% matched to HAM10000_metadata.csv)
- Skin lesion ID grouping & repeated lesion identification
- Ultrasound mask exclusion
- Quarantine manifest integrity & rules
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from metadata_manager.config import (
    MANIFESTS_DIR,
    V2_CANONICAL_COLUMNS,
    DATASET_VERSIONS,
    V2_MODALITY_CLASSES,
)
from metadata_manager.v2_label_reconstructor import V2LabelReconstructor


@pytest.fixture(scope="module")
def manifests():
    """Load all generated manifests into memory."""
    data = {}
    for modality, ver in DATASET_VERSIONS.items():
        csv_p = MANIFESTS_DIR / f"{ver}.csv"
        assert csv_p.exists(), f"Manifest {csv_p} does not exist"
        data[modality] = pd.read_csv(csv_p)
    q_csv = MANIFESTS_DIR / "v2_quarantine.csv"
    assert q_csv.exists(), f"Quarantine manifest {q_csv} does not exist"
    data["quarantine"] = pd.read_csv(q_csv)
    return data


class TestV2CanonicalSchema:
    """Tests for Canonical Metadata Schema compliance."""

    def test_canonical_columns_present(self, manifests):
        for modality, df in manifests.items():
            if modality == "quarantine":
                continue
            assert list(df.columns) == V2_CANONICAL_COLUMNS, f"Schema mismatch in {modality}"

    def test_no_empty_image_paths(self, manifests):
        for modality, df in manifests.items():
            if modality == "quarantine":
                continue
            assert df["image_path"].isnull().sum() == 0, f"Null image_path found in {modality}"
            assert (df["image_path"].str.strip() == "").sum() == 0

    def test_dataset_version_consistency(self, manifests):
        for modality, df in manifests.items():
            if modality == "quarantine":
                continue
            expected_ver = DATASET_VERSIONS[modality]
            assert (df["dataset_version"] == expected_ver).all()


class TestBrainMRIReconstruction:
    """Tests for Brain MRI folder mapping."""

    def test_brain_mri_counts_and_classes(self, manifests):
        df = manifests["brain_mri"]
        assert len(df) == 7200
        expected_classes = {"glioma", "meningioma", "no_tumor", "pituitary"}
        assert set(df["standardized_label"].unique()) == expected_classes
        counts = df["standardized_label"].value_counts().to_dict()
        for cls in expected_classes:
            assert counts[cls] == 1800, f"Class {cls} count is {counts[cls]} != 1800"

    def test_brain_mri_splits_preserved(self, manifests):
        df = manifests["brain_mri"]
        assert (df["split_status"] == "train").sum() == 5600
        assert (df["split_status"] == "test").sum() == 1600


class TestBrainCTReconstruction:
    """Tests for Brain CT CSV-to-image mapping."""

    def test_brain_ct_mapping_and_counts(self, manifests):
        df = manifests["brain_ct"]
        assert len(df) == 200
        assert (df["label_status"] == "VALIDATED").all()
        assert (df["label_type"] == "binary_classification").all()
        counts = df["standardized_label"].value_counts().to_dict()
        assert counts["hemorrhage"] == 100
        assert counts["normal"] == 100

    def test_brain_ct_id_alignment(self, manifests):
        df = manifests["brain_ct"]
        # Check first 5 are hemorrhage and id matches
        first_5 = df.head(5)
        assert (first_5["original_label"] == 1).all()
        assert (first_5["standardized_label"] == "hemorrhage").all()


class TestChestXRayReconstruction:
    """Tests for Chest X-ray class mapping and mask exclusion."""

    def test_chest_xray_classes_and_counts(self, manifests):
        df = manifests["chest_xray"]
        assert len(df) == 21165
        expected_classes = {"covid", "lung_opacity", "normal", "viral_pneumonia"}
        assert set(df["standardized_label"].unique()) == expected_classes
        assert "unknown" not in df["standardized_label"].unique()

    def test_chest_xray_masks_excluded_from_supervised(self, manifests):
        df = manifests["chest_xray"]
        assert not df["image_path"].str.contains("_mask", case=False).any()
        assert not df["image_path"].str.contains("/masks/", case=False).any()


class TestHistopathologyReconstruction:
    """Tests for Histopathology nested folder mapping."""

    def test_histopathology_classes_and_counts(self, manifests):
        df = manifests["histopathology"]
        assert len(df) == 25000
        expected_classes = {
            "colon_adenocarcinoma",
            "colon_normal",
            "lung_adenocarcinoma",
            "lung_normal",
            "lung_squamous_cell_carcinoma",
        }
        assert set(df["standardized_label"].unique()) == expected_classes
        counts = df["standardized_label"].value_counts().to_dict()
        for cls in expected_classes:
            assert counts[cls] == 5000


class TestRetinaReconstruction:
    """Tests for Retina multi-label mapping and quarantine."""

    def test_retina_multi_label_structure(self, manifests):
        df = manifests["retina"]
        assert len(df) == 6392
        assert (df["label_type"] == "multi_label").all()
        assert (df["label_status"] == "VALIDATED").all()

        # Check multi-label vector dimension
        for vec_str in df["multi_label_vector"].head(20):
            vec = json.loads(vec_str)
            assert isinstance(vec, list)
            assert len(vec) == 8
            assert all(v in (0, 1) for v in vec)

    def test_retina_patient_leakage_grouping(self, manifests):
        df = manifests["retina"]
        assert df["patient_id"].notnull().all()
        assert (df["leakage_group_id"] == df["patient_id"]).all()
        # Verify 3358 unique patients
        assert df["patient_id"].nunique() == 3358

    def test_retina_quarantine_exclusions(self, manifests):
        q_df = manifests["quarantine"]
        ret_q = q_df[q_df["modality"] == "retina"]
        reasons = ret_q["quarantine_reason"].value_counts().to_dict()
        assert reasons["Preprocessed fundus crop with undocumented processing provenance and unknown license"] == 6392
        assert reasons["ODIR-5K competition test set image without ground truth labels"] == 1000
        assert reasons["Training image present in folder but omitted from full_df.csv"] == 608


class TestSkinReconstruction:
    """Tests for Skin metadata join and lesion ID tracking."""

    def test_skin_metadata_join(self, manifests):
        df = manifests["skin"]
        assert len(df) == 10015
        assert (df["label_status"] == "VALIDATED").all()
        expected_classes = {
            "melanocytic_nevus",
            "melanoma",
            "benign_keratosis",
            "basal_cell_carcinoma",
            "actinic_keratosis",
            "vascular_lesion",
            "dermatofibroma",
        }
        assert set(df["standardized_label"].unique()) == expected_classes
        counts = df["standardized_label"].value_counts().to_dict()
        assert counts["melanocytic_nevus"] == 6705
        assert counts["melanoma"] == 1113
        assert counts["dermatofibroma"] == 115

    def test_skin_lesion_id_leakage_grouping(self, manifests):
        df = manifests["skin"]
        assert df["lesion_id"].notnull().all()
        assert (df["leakage_group_id"] == df["lesion_id"]).all()
        assert df["lesion_id"].nunique() == 7470
        # 5,514 unique single-image lesions + 4,501 images belonging to repeated lesions
        assert (df["duplicate_status"] == "REPEATED_LESION").sum() == 4501
        assert (df["duplicate_status"] == "UNIQUE").sum() == 5514
        # 2,545 repeat images beyond the 7,470 unique lesions
        assert len(df) - df["lesion_id"].nunique() == 2545



class TestUltrasoundReconstruction:
    """Tests for Ultrasound classification mapping and mask exclusion."""

    def test_ultrasound_classes_and_counts(self, manifests):
        df = manifests["ultrasound"]
        assert len(df) == 780
        expected_classes = {"benign", "malignant", "normal"}
        assert set(df["standardized_label"].unique()) == expected_classes
        counts = df["standardized_label"].value_counts().to_dict()
        assert counts["benign"] == 437
        assert counts["malignant"] == 210
        assert counts["normal"] == 133

    def test_ultrasound_masks_excluded(self, manifests):
        df = manifests["ultrasound"]
        assert not df["image_path"].str.contains("_mask", case=False).any()
        q_df = manifests["quarantine"]
        us_masks = q_df[(q_df["modality"] == "ultrasound") & (q_df["label_status"] == "MASK")]
        assert len(us_masks) == 798


class TestQuarantineRules:
    """Tests for Quarantine manifest integrity."""

    def test_quarantine_total_and_reasons(self, manifests):
        q_df = manifests["quarantine"]
        assert len(q_df) == 47966
        assert q_df["quarantine_reason"].isnull().sum() == 0
        assert set(q_df["label_status"].unique()) == {"MASK", "HOLD", "MISSING_LABEL", "UNKNOWN_SOURCE"}
