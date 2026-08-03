"""
tests/test_registry_generator.py
===================================
Unit tests for metadata_manager.registry_generator.RegistryGenerator.

Tests cover:
  - generate() returns a dict with required top-level keys
  - Total datasets count matches DATASET_REGISTRY_INFO
  - Each dataset entry has required fields
  - save() writes a valid JSON file
  - Error raised if save() called before generate()

Author: Medical AI System — Module 2
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from metadata_manager.registry_generator import RegistryGenerator
from metadata_manager.config import DATASET_REGISTRY_INFO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_master_df() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": ["mri/glioma/img1.jpg", "xray/covid/img2.png"],
        "internal_id": ["MRI_000001", "XRAY_000001"],
        "dataset_name": [
            "COVID-19 Radiography Database",
            "COVID-19 Radiography Database",
        ],
        "dataset_source": ["Kaggle / IEEE DataPort", "Kaggle / IEEE DataPort"],
        "modality": ["chest_xray", "chest_xray"],
        "disease": ["COVID", "COVID"],
        "normalized_class": ["covid", "covid"],
        "original_class": ["COVID", "COVID"],
        "split": ["train", "train"],
        "image_path": ["/data/img1.jpg", "/data/img2.png"],
        "extension": [".jpg", ".png"],
        "width": [224, 224],
        "height": [224, 224],
        "channels": [1, 1],
        "file_size_bytes": [50000, 60000],
    })


SAMPLE_STATS = {
    "generated_at": "2026-01-01T00:00:00+00:00",
    "overall": {"total_images": 2},
}


@pytest.fixture()
def registry_gen() -> RegistryGenerator:
    df = _make_master_df()
    gen = RegistryGenerator(df, SAMPLE_STATS)
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for the generate() method."""

    def test_returns_dict(self, registry_gen):
        result = registry_gen.generate()
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self, registry_gen):
        result = registry_gen.generate()
        for key in ["generated_at", "total_datasets", "total_images", "datasets"]:
            assert key in result, f"Missing key: {key}"

    def test_total_datasets_matches_registry_info(self, registry_gen):
        result = registry_gen.generate()
        assert result["total_datasets"] == len(DATASET_REGISTRY_INFO)

    def test_datasets_is_dict(self, registry_gen):
        result = registry_gen.generate()
        assert isinstance(result["datasets"], dict)

    def test_each_entry_has_required_fields(self, registry_gen):
        result = registry_gen.generate()
        required_fields = {
            "name", "source", "url", "license", "modality",
            "total_images", "valid_images", "num_classes",
            "classes", "last_scan_time",
        }
        for key, entry in result["datasets"].items():
            missing = required_fields - set(entry.keys())
            assert not missing, f"Dataset '{key}' missing fields: {missing}"

    def test_registry_property_after_generate(self, registry_gen):
        assert registry_gen.registry is None
        registry_gen.generate()
        assert registry_gen.registry is not None

    def test_scan_time_from_statistics(self, registry_gen):
        result = registry_gen.generate()
        # Scan time should come from the statistics dict
        for entry in result["datasets"].values():
            assert "last_scan_time" in entry


class TestSave:
    """Tests for the save() method."""

    def test_save_raises_before_generate(self, tmp_path):
        df = _make_master_df()
        gen = RegistryGenerator(df, outputs_dir=tmp_path)
        with pytest.raises(RuntimeError, match="generate"):
            gen.save()

    def test_save_creates_json(self, tmp_path):
        df = _make_master_df()
        gen = RegistryGenerator(df, SAMPLE_STATS, tmp_path)
        gen.generate()
        dest = gen.save()
        assert dest.exists()
        assert dest.suffix == ".json"

    def test_saved_json_is_valid(self, tmp_path):
        df = _make_master_df()
        gen = RegistryGenerator(df, SAMPLE_STATS, tmp_path)
        gen.generate()
        dest = gen.save()
        with open(dest, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "datasets" in data

    def test_saved_json_total_images(self, tmp_path):
        df = _make_master_df()
        gen = RegistryGenerator(df, SAMPLE_STATS, tmp_path)
        gen.generate()
        gen.save()
        with open(tmp_path / "dataset_registry.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_images"] == len(df)
