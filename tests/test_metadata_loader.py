"""
tests/test_metadata_loader.py
===============================
Unit tests for metadata_manager.metadata_loader.MetadataLoader.

Tests cover:
  - Successful loading of all five Module 1 output files
  - Schema validation (correct columns present)
  - FileNotFoundError for missing metadata.csv
  - ValueError for missing required columns
  - is_loaded flag behaviour
  - summary() output

Author: Medical AI System — Module 2
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from metadata_manager.metadata_loader import MetadataLoader, REQUIRED_METADATA_COLS, REQUIRED_LABELS_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metadata_df() -> pd.DataFrame:
    """Minimal valid metadata.csv DataFrame."""
    return pd.DataFrame({
        "image_id": ["brain_mri/glioma/img1.jpg", "chest_xray/COVID-19_Radiography_Dataset/img2.png"],
        "modality": ["brain_mri", "chest_xray"],
        "disease_class": ["glioma", "COVID-19_Radiography_Dataset"],
        "split": ["train", "train"],
        "image_path": ["/data/img1.jpg", "/data/img2.png"],
        "filename": ["img1.jpg", "img2.png"],
        "extension": [".jpg", ".png"],
        "width": [224, 512],
        "height": [224, 512],
        "file_size_bytes": [50000, 120000],
        "is_valid": [True, True],
        "md5_hash": ["aabbcc", "ddeeff"],
    })


def _make_labels_df() -> pd.DataFrame:
    """Minimal valid labels.csv DataFrame."""
    return pd.DataFrame({
        "image_id": ["brain_mri/glioma/img1.jpg", "chest_xray/COVID-19_Radiography_Dataset/img2.png"],
        "modality": ["brain_mri", "chest_xray"],
        "disease_class": ["glioma", "COVID-19_Radiography_Dataset"],
        "split": ["train", "train"],
        "label": [0, 1],
        "class_name": ["glioma", "COVID-19_Radiography_Dataset"],
    })


def _make_distribution_df() -> pd.DataFrame:
    return pd.DataFrame({
        "modality": ["brain_mri"],
        "disease_class": ["glioma"],
        "split": ["train"],
        "count": [1400],
    })


@pytest.fixture()
def valid_outputs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with all five Module 1 output files."""
    _make_metadata_df().to_csv(tmp_path / "metadata.csv", index=False)
    _make_labels_df().to_csv(tmp_path / "labels.csv", index=False)
    _make_distribution_df().to_csv(tmp_path / "class_distribution.csv", index=False)

    stats = {"generated_at": "2026-01-01T00:00:00Z", "overall": {"total_images": 2}}
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(stats), encoding="utf-8")

    report = {"total_images_scanned": 2, "issues": []}
    (tmp_path / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestMetadataLoaderLoad:
    """Tests for the happy-path loading scenario."""

    def test_load_returns_self(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir)
        result = loader.load()
        assert result is loader

    def test_is_loaded_after_load(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir)
        loader.load()
        assert loader.is_loaded is True

    def test_metadata_df_has_correct_shape(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        df = loader.metadata_df
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 12)

    def test_labels_df_loaded(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        assert len(loader.labels_df) == 2

    def test_statistics_loaded(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        assert "overall" in loader.statistics

    def test_validation_report_loaded(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        assert "total_images_scanned" in loader.validation_report

    def test_distribution_df_loaded(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        assert len(loader.distribution_df) == 1


class TestMetadataLoaderErrors:
    """Tests for error handling."""

    def test_missing_metadata_csv_raises_file_not_found(self, tmp_path):
        # Only labels.csv present — no metadata.csv
        _make_labels_df().to_csv(tmp_path / "labels.csv", index=False)
        loader = MetadataLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="metadata.csv"):
            loader.load()

    def test_missing_labels_csv_raises_file_not_found(self, tmp_path):
        _make_metadata_df().to_csv(tmp_path / "metadata.csv", index=False)
        loader = MetadataLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="labels.csv"):
            loader.load()

    def test_bad_schema_metadata_raises_value_error(self, tmp_path):
        # Write metadata.csv with a missing column
        bad_df = _make_metadata_df().drop(columns=["md5_hash"])
        bad_df.to_csv(tmp_path / "metadata.csv", index=False)
        _make_labels_df().to_csv(tmp_path / "labels.csv", index=False)
        loader = MetadataLoader(tmp_path)
        with pytest.raises(ValueError, match="md5_hash"):
            loader.load()

    def test_not_loaded_raises_runtime_error(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir)
        with pytest.raises(RuntimeError, match="load()"):
            _ = loader.metadata_df


class TestMetadataLoaderSummary:
    """Tests for the summary() method."""

    def test_summary_returns_string(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        s = loader.summary()
        assert isinstance(s, str)

    def test_summary_contains_total(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        s = loader.summary()
        assert "2" in s  # 2 images

    def test_summary_contains_modalities(self, valid_outputs_dir):
        loader = MetadataLoader(valid_outputs_dir).load()
        s = loader.summary()
        assert "modalities" in s.lower()
