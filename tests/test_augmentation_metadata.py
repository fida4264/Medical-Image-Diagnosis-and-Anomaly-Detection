"""
tests/test_augmentation_metadata.py
======================================
Unit tests for augmentation.reports.AugmentationMetadataWriter
and AugmentationStatistics.

Run with:
    pytest tests/test_augmentation_metadata.py -v

Author: Medical AI System — Module 4
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from augmentation.reports.metadata_writer import AugmentationMetadataWriter
from augmentation.reports.statistics import AugmentationStatistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer_with_records(n: int = 5):
    writer = AugmentationMetadataWriter()
    writer.set_total_originals(n)
    for i in range(n):
        writer.record(
            original_image_id=f"MRI_{i:06d}",
            original_path=f"/data/processed/brain_mri/glioma/MRI_{i:06d}.png",
            augmented_path=f"/data/augmented/brain_mri/glioma/MRI_{i:06d}_aug000.png",
            modality="brain_mri",
            normalized_class="glioma",
            augmentation_index=0,
            operations_applied=["Rotate", "GaussNoise", "RandomBrightnessContrast"],
            seed=42 + i,
            status="success",
        )
    return writer


# ---------------------------------------------------------------------------
# AugmentationMetadataWriter tests
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_increments_total(self):
        writer = AugmentationMetadataWriter()
        writer.record("ID001", "/a.png", "/b.png", "brain_mri", "glioma", 0, [], 42, "success")
        assert writer.total_records == 1

    def test_multiple_records(self):
        writer = _make_writer_with_records(10)
        assert writer.total_records == 10

    def test_error_record_not_counted_as_augmented(self):
        writer = AugmentationMetadataWriter()
        writer.record("ID001", "/a.png", "", "brain_mri", "glioma", 0, [], 42, "error:io")
        assert writer.total_augmented == 0
        assert writer.total_errors == 1

    def test_success_record_counted_as_augmented(self):
        writer = AugmentationMetadataWriter()
        writer.record("ID001", "/a.png", "/b.png", "brain_mri", "glioma", 0, [], 42, "success")
        assert writer.total_augmented == 1


class TestSaveCSV:
    def test_save_csv_creates_file(self, tmp_path):
        writer = _make_writer_with_records(3)
        path = writer.save_csv(tmp_path / "augmentation_metadata.csv")
        assert path.exists()

    def test_saved_csv_is_valid(self, tmp_path):
        writer = _make_writer_with_records(3)
        path = writer.save_csv(tmp_path / "augmentation_metadata.csv")
        df = pd.read_csv(path)
        assert len(df) == 3

    def test_csv_has_required_columns(self, tmp_path):
        writer = _make_writer_with_records(3)
        path = writer.save_csv(tmp_path / "aug_meta.csv")
        df = pd.read_csv(path)
        required = [
            "original_image_id", "original_path", "augmented_path",
            "modality", "normalized_class", "augmentation_index",
            "operations_applied", "seed", "timestamp", "status",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_creates_parent_directory(self, tmp_path):
        writer = _make_writer_with_records(2)
        nested = tmp_path / "sub" / "dir" / "aug.csv"
        writer.save_csv(nested)
        assert nested.exists()

    def test_operations_joined_with_pipe(self, tmp_path):
        writer = AugmentationMetadataWriter()
        writer.record("ID1", "/a.png", "/b.png", "skin", "benign", 0,
                       ["Rotate", "HorizontalFlip", "GaussNoise"], 42, "success")
        path = writer.save_csv(tmp_path / "meta.csv")
        df = pd.read_csv(path)
        assert "Rotate|HorizontalFlip|GaussNoise" == df["operations_applied"].iloc[0]


class TestSaveReport:
    def test_save_report_creates_file(self, tmp_path):
        writer = _make_writer_with_records(5)
        path = writer.save_report(tmp_path / "report.json")
        assert path.exists()

    def test_report_is_valid_json(self, tmp_path):
        writer = _make_writer_with_records(5)
        path = writer.save_report(tmp_path / "report.json")
        with open(path) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_report_has_required_keys(self, tmp_path):
        writer = _make_writer_with_records(5)
        path = writer.save_report(tmp_path / "report.json")
        data = json.loads(path.read_text())
        for key in [
            "generated_at", "total_original_images", "total_augmented_images",
            "augmentation_factor", "by_modality",
        ]:
            assert key in data, f"Missing key: {key}"

    def test_total_augmented_correct(self, tmp_path):
        writer = _make_writer_with_records(5)
        path = writer.save_report(tmp_path / "report.json")
        data = json.loads(path.read_text())
        assert data["total_augmented_images"] == 5


# ---------------------------------------------------------------------------
# AugmentationStatistics tests
# ---------------------------------------------------------------------------

@pytest.fixture
def orig_df():
    return pd.DataFrame([
        {"modality": "brain_mri", "normalized_class": "glioma"},
        {"modality": "brain_mri", "normalized_class": "glioma"},
        {"modality": "brain_mri", "normalized_class": "notumor"},
        {"modality": "chest_xray", "normalized_class": "normal"},
        {"modality": "chest_xray", "normalized_class": "covid"},
    ])


@pytest.fixture
def aug_df():
    return pd.DataFrame([
        {"modality": "brain_mri", "normalized_class": "glioma", "status": "success"},
        {"modality": "brain_mri", "normalized_class": "notumor", "status": "success"},
        {"modality": "chest_xray", "normalized_class": "normal", "status": "success"},
    ])


class TestAugmentationStatistics:
    def test_compute_returns_dict(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        result = stats.compute()
        assert isinstance(result, dict)

    def test_has_required_keys(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        result = stats.compute()
        for key in [
            "generated_at", "total_original_images", "total_augmented_images",
            "before_augmentation", "after_augmentation",
        ]:
            assert key in result

    def test_total_original_count(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        result = stats.compute()
        assert result["total_original_images"] == 5

    def test_total_augmented_count(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        result = stats.compute()
        assert result["total_augmented_images"] == 3

    def test_before_modalities_present(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        result = stats.compute()
        assert "brain_mri" in result["before_augmentation"]
        assert "chest_xray" in result["before_augmentation"]

    def test_raises_before_load(self):
        stats = AugmentationStatistics()
        with pytest.raises(RuntimeError):
            stats.compute()

    def test_save_raises_before_compute(self, tmp_path):
        stats = AugmentationStatistics()
        with pytest.raises(RuntimeError):
            stats.save(tmp_path / "stats.json")

    def test_save_creates_file(self, orig_df, aug_df, tmp_path):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        stats.compute()
        path = stats.save(tmp_path / "stats.json")
        assert path.exists()

    def test_statistics_property_none_before_compute(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        assert stats.statistics is None

    def test_statistics_property_after_compute(self, orig_df, aug_df):
        stats = AugmentationStatistics()
        stats.load_original(orig_df)
        stats.load_augmented(aug_df)
        stats.compute()
        assert stats.statistics is not None
