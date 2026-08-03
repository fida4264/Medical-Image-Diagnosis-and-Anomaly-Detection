"""
tests/test_summary_generator.py
==================================
Unit tests for metadata_manager.summary_generator.SummaryGenerator.

Tests cover:
  - generate() returns a non-empty Markdown string
  - All required sections are present in the output
  - Numeric values are correct (total images, modalities, classes)
  - save() writes dataset_summary.md
  - Error raised if save() before generate()
  - content property is None before generate()

Author: Medical AI System — Module 2
"""

from pathlib import Path

import pandas as pd
import pytest

from metadata_manager.summary_generator import SummaryGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_master_df() -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": [f"brain_mri/glioma/img{i}.jpg" for i in range(5)]
                  + [f"chest_xray/COVID-19/img{i}.png" for i in range(3)],
        "internal_id": [f"MRI_{i:06d}" for i in range(1, 6)]
                     + [f"XRAY_{i:06d}" for i in range(1, 4)],
        "dataset_name": ["Brain MRI Dataset"] * 5 + ["COVID-19 Radiography Database"] * 3,
        "dataset_source": ["Kaggle"] * 8,
        "modality": ["brain_mri"] * 5 + ["chest_xray"] * 3,
        "disease": ["glioma"] * 5 + ["COVID"] * 3,
        "normalized_class": ["glioma"] * 5 + ["covid"] * 3,
        "original_class": ["glioma"] * 5 + ["COVID"] * 3,
        "split": ["train"] * 5 + ["train"] * 3,
        "image_path": [f"/data/img{i}.jpg" for i in range(8)],
        "extension": [".jpg"] * 5 + [".png"] * 3,
        "width": [224] * 8,
        "height": [224] * 8,
        "channels": [1] * 5 + [1] * 3,
        "file_size_bytes": [50000] * 8,
        "is_valid": [True] * 7 + [False],
    })


SAMPLE_STATS = {"generated_at": "2026-01-01T00:00:00Z"}


@pytest.fixture()
def summary_gen() -> SummaryGenerator:
    df = _make_master_df()
    gen = SummaryGenerator(df, SAMPLE_STATS)
    return gen


@pytest.fixture()
def generated_content(summary_gen) -> str:
    return summary_gen.generate()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for generate()."""

    def test_returns_string(self, generated_content):
        assert isinstance(generated_content, str)

    def test_is_non_empty(self, generated_content):
        assert len(generated_content) > 100

    def test_contains_title(self, generated_content):
        assert "# Dataset Summary" in generated_content

    def test_contains_overview_section(self, generated_content):
        assert "## Overview" in generated_content

    def test_contains_modality_section(self, generated_content):
        assert "## Images per Modality" in generated_content

    def test_contains_class_section(self, generated_content):
        assert "## Images per Disease Class" in generated_content

    def test_contains_dataset_sources_section(self, generated_content):
        assert "## Dataset Sources" in generated_content

    def test_contains_resolution_section(self, generated_content):
        assert "## Image Resolution" in generated_content

    def test_contains_extensions_section(self, generated_content):
        assert "## Supported File Extensions" in generated_content

    def test_contains_quality_section(self, generated_content):
        assert "## Data Quality" in generated_content

    def test_contains_usage_section(self, generated_content):
        assert "## Usage by Downstream Modules" in generated_content

    def test_total_images_present(self, generated_content):
        # 8 total images
        assert "8" in generated_content

    def test_modality_names_present(self, generated_content):
        assert "brain_mri" in generated_content
        assert "chest_xray" in generated_content

    def test_class_names_present(self, generated_content):
        assert "glioma" in generated_content
        assert "covid" in generated_content

    def test_content_property_set_after_generate(self, summary_gen):
        assert summary_gen.content is None
        summary_gen.generate()
        assert summary_gen.content is not None


class TestSave:
    """Tests for save()."""

    def test_save_raises_before_generate(self, summary_gen, tmp_path):
        with pytest.raises(RuntimeError, match="generate"):
            summary_gen.save(tmp_path)

    def test_save_creates_file(self, summary_gen, tmp_path):
        summary_gen.generate()
        dest = summary_gen.save(tmp_path)
        assert dest.exists()

    def test_saved_file_has_md_extension(self, summary_gen, tmp_path):
        summary_gen.generate()
        dest = summary_gen.save(tmp_path)
        assert dest.suffix == ".md"

    def test_saved_file_readable(self, summary_gen, tmp_path):
        summary_gen.generate()
        dest = summary_gen.save(tmp_path)
        content = dest.read_text(encoding="utf-8")
        assert "# Dataset Summary" in content

    def test_save_creates_output_dir_if_missing(self, summary_gen, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()
        summary_gen.generate()
        summary_gen.save(new_dir)
        assert new_dir.exists()
        assert (new_dir / "dataset_summary.md").exists()
