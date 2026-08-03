"""
tests/test_image_mapper.py
============================
Unit tests for metadata_manager.image_mapper.ImageMapper.

Tests cover:
  - generate_ids() produces the correct format (PREFIX_NNNNNN)
  - IDs are globally unique
  - Sequence resets per modality prefix
  - mapping_df has required columns
  - save() writes image_mapping.csv
  - lookup_original() returns correct path
  - Error raised if save() called before generate_ids()

Author: Medical AI System — Module 2
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from metadata_manager.image_mapper import ImageMapper
from metadata_manager.config import COL_INTERNAL_ID, COL_IMAGE_ID, COL_MODALITY, COL_IMAGE_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_master_df(n_mri: int = 3, n_xray: int = 2) -> pd.DataFrame:
    """Create a minimal master_metadata DataFrame for testing."""
    rows = []
    for i in range(n_mri):
        rows.append({
            COL_IMAGE_ID: f"brain_mri/glioma/img{i}.jpg",
            COL_MODALITY: "brain_mri",
            COL_IMAGE_PATH: f"D:/datasets/brain_mri/glioma/img{i}.jpg",
            "extension": ".jpg",
        })
    for i in range(n_xray):
        rows.append({
            COL_IMAGE_ID: f"chest_xray/COVID-19/img{i}.png",
            COL_MODALITY: "chest_xray",
            COL_IMAGE_PATH: f"D:/datasets/chest_xray/COVID-19/img{i}.png",
            "extension": ".png",
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def small_master_df() -> pd.DataFrame:
    return _make_master_df(n_mri=3, n_xray=2)


@pytest.fixture()
def mapper(small_master_df) -> ImageMapper:
    m = ImageMapper(small_master_df)
    m.generate_ids()
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateIds:
    """Tests for generate_ids()."""

    def test_returns_series(self, small_master_df):
        m = ImageMapper(small_master_df)
        result = m.generate_ids()
        assert isinstance(result, pd.Series)

    def test_series_length_matches_df(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids()
        assert len(ids) == len(small_master_df)

    def test_ids_are_unique(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids()
        assert ids.nunique() == len(ids)

    def test_id_format_mri(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids()
        mri_ids = ids[small_master_df[COL_MODALITY] == "brain_mri"]
        for iid in mri_ids:
            assert re.match(r"^MRI_\d{6}$", iid), f"Bad ID format: {iid}"

    def test_id_format_xray(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids()
        xray_ids = ids[small_master_df[COL_MODALITY] == "chest_xray"]
        for iid in xray_ids:
            assert re.match(r"^XRAY_\d{6}$", iid), f"Bad ID format: {iid}"

    def test_sequence_starts_at_1(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids().tolist()
        assert "MRI_000001" in ids
        assert "XRAY_000001" in ids

    def test_mri_sequence_is_sequential(self, small_master_df):
        m = ImageMapper(small_master_df)
        ids = m.generate_ids()
        mri_ids = sorted(ids[small_master_df[COL_MODALITY] == "brain_mri"].tolist())
        expected = [f"MRI_{i:06d}" for i in range(1, 4)]
        assert mri_ids == expected


class TestMappingDf:
    """Tests for the mapping_df property."""

    def test_mapping_df_is_none_before_generate(self, small_master_df):
        m = ImageMapper(small_master_df)
        assert m.mapping_df is None

    def test_mapping_df_has_required_columns(self, mapper):
        df = mapper.mapping_df
        assert df is not None
        for col in [COL_INTERNAL_ID, COL_IMAGE_ID, "original_filename", "original_path", "new_internal_name"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_mapping_df_correct_length(self, mapper, small_master_df):
        assert len(mapper.mapping_df) == len(small_master_df)


class TestSave:
    """Tests for save()."""

    def test_save_raises_before_generate(self, small_master_df, tmp_path):
        m = ImageMapper(small_master_df, tmp_path)
        with pytest.raises(RuntimeError, match="generate_ids"):
            m.save()

    def test_save_creates_csv(self, mapper, tmp_path):
        dest = mapper.save(tmp_path)
        assert dest.exists()

    def test_saved_csv_has_correct_columns(self, mapper, tmp_path):
        mapper.save(tmp_path)
        df = pd.read_csv(tmp_path / "image_mapping.csv")
        assert COL_INTERNAL_ID in df.columns
        assert "original_path" in df.columns

    def test_saved_csv_row_count(self, small_master_df, tmp_path):
        m = ImageMapper(small_master_df, tmp_path)
        m.generate_ids()
        m.save(tmp_path)
        df = pd.read_csv(tmp_path / "image_mapping.csv")
        assert len(df) == len(small_master_df)


class TestLookupOriginal:
    """Tests for lookup_original()."""

    def test_lookup_returns_path(self, mapper):
        first_id = mapper.id_series.iloc[0]
        path = mapper.lookup_original(first_id)
        assert path is not None
        assert "img" in path

    def test_lookup_unknown_id_returns_none(self, mapper):
        result = mapper.lookup_original("MRI_999999")
        assert result is None

    def test_lookup_before_generate_returns_none(self, small_master_df):
        m = ImageMapper(small_master_df)
        assert m.lookup_original("MRI_000001") is None
