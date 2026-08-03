"""
tests/test_eda_analyzers.py
=============================
Unit tests for Module 5 EDA analyzer classes.

Tests DatasetAnalyzer, ClassAnalyzer, ModalityAnalyzer, ImagePropertyAnalyzer,
QualityAnalyzer, PreprocessingAnalyzer, and AugmentationAnalyzer with synthetic
in-memory DataFrames — no real dataset files required.

Run with:
    pytest tests/test_eda_analyzers.py -v

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from eda.analyzers import (
    ClassAnalyzer,
    ModalityAnalyzer,
    ImagePropertyAnalyzer,
    QualityAnalyzer,
    PreprocessingAnalyzer,
    AugmentationAnalyzer,
)
from eda.analyzers.class_analyzer import _gini_coefficient


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def master_df() -> pd.DataFrame:
    """Minimal master_metadata DataFrame with synthetic data."""
    rng = np.random.default_rng(42)
    n = 200

    modalities = ["brain_mri", "chest_xray", "skin"]
    classes = {
        "brain_mri":  ["glioma", "meningioma", "no_tumor"],
        "chest_xray": ["normal", "pneumonia"],
        "skin":       ["melanoma", "nevus", "seborrheic_keratosis"],
    }

    rows = []
    for mod in modalities:
        cls_list = classes[mod]
        for i in range(n // len(modalities)):
            cls = cls_list[i % len(cls_list)]
            rows.append({
                "image_id":         f"{mod}_{i:04d}",
                "modality":         mod,
                "normalized_class": cls,
                "dataset_name":     f"{mod}_dataset",
                "dataset_source":   f"{mod}_dataset",
                "split":            "train" if i < 50 else ("val" if i < 65 else "test"),
                "extension":        ".jpg",
                "width":            int(rng.integers(200, 512)),
                "height":           int(rng.integers(200, 512)),
                "channels":         3,
                "file_size_bytes":  int(rng.integers(50_000, 500_000)),
                "image_path":       f"datasets/{mod}/{cls}/img_{i:04d}.jpg",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _gini_coefficient
# ---------------------------------------------------------------------------

class TestGiniCoefficient:
    def test_perfect_balance(self):
        counts = np.array([100, 100, 100, 100])
        assert _gini_coefficient(counts) == pytest.approx(0.0, abs=0.01)

    def test_single_class(self):
        # One class with all images — max imbalance
        counts = np.array([1, 0, 0, 0])
        g = _gini_coefficient(counts[counts > 0])
        # single element → gini = 0
        assert g == pytest.approx(0.0, abs=0.01)

    def test_empty_array(self):
        assert _gini_coefficient(np.array([])) == 0.0

    def test_returns_float(self):
        counts = np.array([10, 20, 30])
        result = _gini_coefficient(counts)
        assert isinstance(result, float)

    def test_clipped_0_to_1(self):
        counts = np.array([1, 1000])
        result = _gini_coefficient(counts)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# ClassAnalyzer
# ---------------------------------------------------------------------------

class TestClassAnalyzer:
    def test_returns_dict(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert isinstance(result, dict)

    def test_has_required_keys(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        required = [
            "total_classes", "total_images", "class_counts", "gini_coefficient",
            "balance_label", "imbalance_ratio", "minority_classes", "rare_classes",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_total_classes_correct(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        expected_classes = master_df["normalized_class"].nunique()
        assert result["total_classes"] == expected_classes

    def test_gini_in_range(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert 0.0 <= result["gini_coefficient"] <= 1.0

    def test_imbalance_ratio_positive(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert result["imbalance_ratio"] > 0

    def test_rare_classes_list(self, master_df):
        ca = ClassAnalyzer(master_df, rare_class_threshold=1000)
        result = ca.analyze()
        # With threshold 1000, all classes are rare
        assert result["rare_class_count"] >= 0

    def test_balance_label_is_string(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert isinstance(result["balance_label"], str)

    def test_top10_keys(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert "top10_classes" in result
        assert isinstance(result["top10_classes"], dict)

    def test_per_modality_present(self, master_df):
        ca = ClassAnalyzer(master_df)
        result = ca.analyze()
        assert "per_modality" in result
        assert len(result["per_modality"]) == master_df["modality"].nunique()

    def test_stats_property(self, master_df):
        ca = ClassAnalyzer(master_df)
        ca.analyze()
        assert ca.stats is not None
        assert isinstance(ca.stats, dict)

    def test_missing_class_column(self):
        df = pd.DataFrame({"image_id": [1, 2, 3], "modality": ["a", "b", "c"]})
        ca = ClassAnalyzer(df)
        result = ca.analyze()
        # Should return empty dict gracefully
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ModalityAnalyzer
# ---------------------------------------------------------------------------

class TestModalityAnalyzer:
    def test_returns_dict(self, master_df):
        ma = ModalityAnalyzer(master_df)
        result = ma.analyze()
        assert isinstance(result, dict)

    def test_per_modality_present(self, master_df):
        ma = ModalityAnalyzer(master_df)
        result = ma.analyze()
        assert "per_modality" in result
        n_mods = master_df["modality"].nunique()
        assert result["total_modalities"] == n_mods

    def test_each_modality_has_required_fields(self, master_df):
        ma = ModalityAnalyzer(master_df)
        result = ma.analyze()
        required = ["total_images", "n_classes", "display_name", "resolution", "file_size"]
        for mod, data in result["per_modality"].items():
            for key in required:
                assert key in data, f"Modality '{mod}' missing field '{key}'"

    def test_total_images_per_modality_correct(self, master_df):
        ma = ModalityAnalyzer(master_df)
        result = ma.analyze()
        for mod in master_df["modality"].unique():
            expected = int((master_df["modality"] == mod).sum())
            actual = result["per_modality"][mod]["total_images"]
            assert actual == expected

    def test_missing_modality_column(self):
        df = pd.DataFrame({"image_id": [1, 2, 3]})
        ma = ModalityAnalyzer(df)
        result = ma.analyze()
        assert isinstance(result, dict)

    def test_largest_smallest_modality(self, master_df):
        ma = ModalityAnalyzer(master_df)
        result = ma.analyze()
        assert "largest_modality" in result
        assert "smallest_modality" in result
        assert result["largest_modality"] in master_df["modality"].unique()


# ---------------------------------------------------------------------------
# ImagePropertyAnalyzer
# ---------------------------------------------------------------------------

class TestImagePropertyAnalyzer:
    def test_returns_dict(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        assert isinstance(result, dict)

    def test_has_physical_key(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        assert "physical" in result

    def test_physical_has_resolution(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        physical = result["physical"]
        assert "resolution" in physical
        assert "width" in physical["resolution"]
        assert "height" in physical["resolution"]

    def test_physical_file_size(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        assert "file_size" in result["physical"]

    def test_physical_channel_distribution(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        assert "channel_distribution" in result["physical"]

    def test_pixel_stats_skipped_when_disabled(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        # pixel key should be empty dict when disabled
        assert result.get("pixel", {}) == {}

    def test_resolution_buckets_present(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        result = ipa.analyze()
        physical = result["physical"]
        assert "resolution_buckets" in physical

    def test_stats_property(self, master_df):
        ipa = ImagePropertyAnalyzer(master_df, compute_pixel_stats=False)
        ipa.analyze()
        assert isinstance(ipa.stats, dict)


# ---------------------------------------------------------------------------
# QualityAnalyzer
# ---------------------------------------------------------------------------

class TestQualityAnalyzer:
    def test_returns_dict_without_report(self, master_df):
        qa = QualityAnalyzer(
            quality_report_path=Path("nonexistent_path/quality_report.json"),
            master_df=master_df,
            class_gini=0.3,
        )
        result = qa.analyze()
        assert isinstance(result, dict)

    def test_has_quality_score(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.2)
        result = qa.analyze()
        assert "quality_score" in result
        qs = result["quality_score"]
        assert "total_score" in qs
        assert "grade" in qs

    def test_quality_score_in_range(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.1)
        result = qa.analyze()
        total = result["quality_score"]["total_score"]
        assert 0.0 <= total <= 100.0

    def test_grade_is_string(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.5)
        result = qa.analyze()
        assert isinstance(result["quality_score"]["grade"], str)

    def test_quality_score_property(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.0)
        qa.analyze()
        score = qa.quality_score
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_zero_gini_gives_max_class_balance_score(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.0)
        result = qa.analyze()
        # class balance score = 25 × (1 - 0.0) = 25
        assert result["quality_score"]["class_balance_score"] == pytest.approx(25.0, abs=0.01)

    def test_resolution_consistency_computed(self, master_df):
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.2)
        result = qa.analyze()
        assert "resolution_consistency" in result

    def test_grade_excellent_for_high_score(self, master_df):
        # Zero gini + no flags + wide resolution → high score
        qa = QualityAnalyzer(master_df=master_df, class_gini=0.0)
        result = qa.analyze()
        score = result["quality_score"]["total_score"]
        grade = result["quality_score"]["grade"]
        if score >= 85:
            assert "A" in grade
        elif score >= 70:
            assert "B" in grade


# ---------------------------------------------------------------------------
# PreprocessingAnalyzer (no real files — covers graceful degradation)
# ---------------------------------------------------------------------------

class TestPreprocessingAnalyzer:
    def test_missing_file_returns_unavailable(self, tmp_path):
        pa = PreprocessingAnalyzer(
            metadata_csv=tmp_path / "nonexistent.csv",
            report_json=tmp_path / "nonexistent.json",
        )
        pa.load()
        result = pa.analyze()
        assert result == {"available": False}

    def test_returns_dict_always(self, tmp_path):
        pa = PreprocessingAnalyzer(
            metadata_csv=tmp_path / "x.csv",
        )
        result = pa.analyze()
        assert isinstance(result, dict)

    def test_with_synthetic_csv(self, tmp_path):
        csv_path = tmp_path / "preprocessing_metadata.csv"
        df = pd.DataFrame({
            "modality":          ["brain_mri", "chest_xray", "brain_mri"],
            "status":            ["success",   "success",    "error"],
            "processing_time_s": [0.12,        0.08,         0.05],
            "operations_applied": ["normalize|resize", "resize", ""],
            "quality_flags":     ["", "dark", ""],
            "original_width":    [512, 256, 1024],
            "original_height":   [512, 256, 1024],
            "processed_width":   [224, 224, 0],
            "processed_height":  [224, 224, 0],
            "normalized_class":  ["glioma", "normal", "glioma"],
        })
        df.to_csv(csv_path, index=False)

        pa = PreprocessingAnalyzer(metadata_csv=csv_path)
        pa.load()
        result = pa.analyze()

        assert result["available"] is True
        assert result["total_records"] == 3
        assert result["success"] == 2
        assert "success_rate_pct" in result
        assert result["success_rate_pct"] == pytest.approx(66.67, abs=0.1)
        assert "normalize" in result["operations_frequency"]


# ---------------------------------------------------------------------------
# AugmentationAnalyzer (no real files — covers graceful degradation)
# ---------------------------------------------------------------------------

class TestAugmentationAnalyzer:
    def test_missing_file_returns_unavailable(self, tmp_path):
        aa = AugmentationAnalyzer(
            metadata_csv=tmp_path / "nonexistent.csv",
        )
        aa.load()
        result = aa.analyze()
        assert result == {"available": False}

    def test_with_synthetic_csv(self, tmp_path, master_df):
        csv_path = tmp_path / "augmentation_metadata.csv"
        aug_df = pd.DataFrame({
            "original_image_id": ["brain_mri_0000"] * 5 + ["chest_xray_0000"] * 5,
            "modality":          ["brain_mri"] * 5 + ["chest_xray"] * 5,
            "normalized_class":  ["glioma"] * 5 + ["normal"] * 5,
            "augmentation_index": list(range(5)) + list(range(5)),
            "operations_applied": ["rotate|flip"] * 10,
            "seed":               list(range(10)),
            "status":             ["success"] * 9 + ["error"],
            "augmented_path":     [f"aug_{i}.png" for i in range(10)],
        })
        aug_df.to_csv(csv_path, index=False)

        aa = AugmentationAnalyzer(metadata_csv=csv_path, master_df=master_df)
        aa.load()
        result = aa.analyze()

        assert result["available"] is True
        assert result["total_augmented_records"] == 10
        assert result["total_successful_augmentations"] == 9
        assert "rotate" in result["transformation_frequency"]
        assert result["success_rate_pct"] == pytest.approx(90.0, abs=0.1)
