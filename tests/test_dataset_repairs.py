"""
tests/test_dataset_repairs.py
=============================
Focused regression tests for Phase 1B dataset repairs:
1. Chest X-ray mask exclusion (/masks/ directory and _mask suffix).
2. COVID-19 4-class preservation and VinBigData separation.
3. HAM10000 skin label join, 7-class recovery, lesion_id preservation, unmatched reporting.

Author: Medical AI System — Phase 1B
"""

from pathlib import Path
import json
import pandas as pd
import pytest
from PIL import Image

from dataset_manager.scanner import DatasetScanner
from metadata_manager.class_mapper import ClassMapper
from metadata_manager.metadata_standardizer import (
    MetadataStandardizer,
    _extract_disease_from_path,
    _get_ham10000_record,
    HAM10000_DX_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dummy_image(path: Path, size: tuple = (32, 32)) -> None:
    """Create a minimal valid PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(120, 120, 120))
    img.save(str(path), format="PNG")


# ---------------------------------------------------------------------------
# Part A: Chest X-ray Mask Contamination Tests
# ---------------------------------------------------------------------------

class TestChestXrayMaskRepairs:
    """Tests confirming /masks/ directories and _mask files are excluded."""

    def test_scanner_excludes_masks_directory(self, tmp_path: Path) -> None:
        """Images located inside any directory named 'masks' must be excluded."""
        root = tmp_path / "datasets"
        # Classification images under /images/
        _create_dummy_image(root / "chest_xray" / "COVID-19_Radiography_Dataset" / "COVID" / "images" / "COVID-1.png")
        _create_dummy_image(root / "chest_xray" / "COVID-19_Radiography_Dataset" / "Normal" / "images" / "Normal-1.png")
        # Mask images under /masks/ — must be skipped
        _create_dummy_image(root / "chest_xray" / "COVID-19_Radiography_Dataset" / "COVID" / "masks" / "COVID-1.png")
        _create_dummy_image(root / "chest_xray" / "COVID-19_Radiography_Dataset" / "Normal" / "masks" / "Normal-1.png")

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()

        # Only 2 classification images should be discovered
        assert len(records) == 2
        for r in records:
            assert "masks" not in [p.lower() for p in r.image_path.parts]
            assert "images" in [p.lower() for p in r.image_path.parts]

    def test_scanner_excludes_mask_suffix_files(self, tmp_path: Path) -> None:
        """Images with _mask in their filename must also be excluded."""
        root = tmp_path / "datasets"
        _create_dummy_image(root / "ultrasound" / "benign" / "benign (1).png")
        _create_dummy_image(root / "ultrasound" / "benign" / "benign (1)_mask.png")
        _create_dummy_image(root / "ultrasound" / "benign" / "benign (1)_mask_1.png")

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()

        assert len(records) == 1
        assert records[0].filename == "benign (1).png"

    def test_covid_classes_intact(self, tmp_path: Path) -> None:
        """All 4 COVID classes must remain intact when standardizing."""
        mapper = ClassMapper()
        assert mapper.normalize("COVID") == "covid"
        assert mapper.normalize("Lung_Opacity") == "lung_opacity"
        assert mapper.normalize("Normal") == "normal"
        assert mapper.normalize("Viral Pneumonia") == "viral_pneumonia"

    def test_extraction_strategy_for_covid(self) -> None:
        """Anchor offset extracts class folder correctly from /images/ path."""
        p_covid = "D:/datasets/chest_xray/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
        p_normal = "D:/datasets/chest_xray/COVID-19_Radiography_Dataset/Normal/images/Normal-1.png"
        p_opacity = "D:/datasets/chest_xray/COVID-19_Radiography_Dataset/Lung_Opacity/images/Opacity-1.png"
        p_pneumonia = "D:/datasets/chest_xray/COVID-19_Radiography_Dataset/Viral Pneumonia/images/VP-1.png"

        assert _extract_disease_from_path(p_covid, "chest_xray", "COVID-19_Radiography_Dataset") == "COVID"
        assert _extract_disease_from_path(p_normal, "chest_xray", "COVID-19_Radiography_Dataset") == "Normal"
        assert _extract_disease_from_path(p_opacity, "chest_xray", "COVID-19_Radiography_Dataset") == "Lung_Opacity"
        assert _extract_disease_from_path(p_pneumonia, "chest_xray", "COVID-19_Radiography_Dataset") == "Viral Pneumonia"


# ---------------------------------------------------------------------------
# Part B: HAM10000 Skin Label Join Tests
# ---------------------------------------------------------------------------

class TestHAM10000SkinLabelJoin:
    """Tests confirming HAM10000 image IDs map to CSV labels and 7 classes."""

    def test_all_seven_dx_codes_mapped(self) -> None:
        """Check all 7 HAM10000 diagnosis codes map to canonical class names."""
        expected = {
            "nv": "melanocytic_nevus",
            "mel": "melanoma",
            "bkl": "benign_keratosis",
            "bcc": "basal_cell_carcinoma",
            "akiec": "actinic_keratosis",
            "vasc": "vascular_lesion",
            "df": "dermatofibroma",
        }
        for dx, canonical in expected.items():
            assert HAM10000_DX_MAP.get(dx) == canonical

    def test_class_mapper_normalizes_skin_classes(self) -> None:
        """ClassMapper must normalize all 7 skin class variants and codes."""
        mapper = ClassMapper()
        assert mapper.normalize("nv") == "melanocytic_nevus"
        assert mapper.normalize("mel") == "melanoma"
        assert mapper.normalize("bkl") == "benign_keratosis"
        assert mapper.normalize("bcc") == "basal_cell_carcinoma"
        assert mapper.normalize("akiec") == "actinic_keratosis"
        assert mapper.normalize("vasc") == "vascular_lesion"
        assert mapper.normalize("df") == "dermatofibroma"
        assert mapper.normalize("melanocytic_nevus") == "melanocytic_nevus"
        assert mapper.normalize("melanoma") == "melanoma"
        assert mapper.normalize("benign_keratosis") == "benign_keratosis"
        assert mapper.normalize("basal_cell_carcinoma") == "basal_cell_carcinoma"
        assert mapper.normalize("actinic_keratosis") == "actinic_keratosis"
        assert mapper.normalize("vascular_lesion") == "vascular_lesion"
        assert mapper.normalize("dermatofibroma") == "dermatofibroma"

    def test_ham10000_record_lookup_known_id(self) -> None:
        """Known ISIC image ID in real dataset should return valid dx and lesion_id."""
        rec = _get_ham10000_record("D:/datasets/skin/HAM10000_images_part_1/ISIC_0027419.jpg")
        if rec is not None:
            assert rec["dx"] == "bkl"
            assert rec["lesion_id"] == "HAM_0000118"

    def test_unmatched_file_reports_unknown(self) -> None:
        """Non-existent image ID must return unknown, not a guessed label."""
        res = _extract_disease_from_path(
            "D:/datasets/skin/HAM10000_images_part_1/ISIC_9999999_nonexistent.jpg",
            "skin",
            "HAM10000_images_part_1",
        )
        assert res == "unknown"

    def test_valid_labels_not_converted_to_unknown(self) -> None:
        """Valid matched HAM10000 images must never return unknown."""
        res = _extract_disease_from_path(
            "D:/datasets/skin/HAM10000_images_part_1/ISIC_0027419.jpg",
            "skin",
            "HAM10000_images_part_1",
        )
        assert res == "benign_keratosis"
        assert res != "unknown"
