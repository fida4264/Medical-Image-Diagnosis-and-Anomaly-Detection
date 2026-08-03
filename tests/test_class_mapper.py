"""
tests/test_class_mapper.py
============================
Unit tests for metadata_manager.class_mapper.ClassMapper.

Tests cover:
  - Exact match normalisation for all known class variants
  - Case-insensitive matching
  - Whitespace / underscore / hyphen collapsing
  - Unknown classes return 'unknown'
  - build_mapping() produces correct DataFrame
  - apply() works on a pandas Series
  - save() writes the CSV file

Author: Medical AI System — Module 2
"""

from pathlib import Path

import pandas as pd
import pytest

from metadata_manager.class_mapper import ClassMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mapper() -> ClassMapper:
    return ClassMapper()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalize:
    """Tests for the normalize() method."""

    # --- COVID variants ---
    @pytest.mark.parametrize("raw,expected", [
        ("COVID", "covid"),
        ("COVID-19", "covid"),
        ("Covid19", "covid"),
        ("covid", "covid"),
        ("covid_19", "covid"),
    ])
    def test_covid_variants(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Lung Opacity ---
    @pytest.mark.parametrize("raw,expected", [
        ("Lung_Opacity", "lung_opacity"),
        ("Lung Opacity", "lung_opacity"),
        ("lungopacity", "lung_opacity"),
        ("lung-opacity", "lung_opacity"),
    ])
    def test_lung_opacity_variants(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Viral Pneumonia ---
    @pytest.mark.parametrize("raw,expected", [
        ("Viral Pneumonia", "viral_pneumonia"),
        ("viral_pneumonia", "viral_pneumonia"),
        ("viralpneumonia", "viral_pneumonia"),
        ("viral-pneumonia", "viral_pneumonia"),
    ])
    def test_viral_pneumonia_variants(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Normal ---
    @pytest.mark.parametrize("raw,expected", [
        ("Normal", "normal"),
        ("NORMAL", "normal"),
        ("normal", "normal"),
        ("no finding", "normal"),
    ])
    def test_normal_variants(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Brain MRI ---
    @pytest.mark.parametrize("raw,expected", [
        ("glioma", "glioma"),
        ("meningioma", "meningioma"),
        ("notumor", "notumor"),
        ("No Tumor", "notumor"),
        ("No_Tumor", "notumor"),
        ("pituitary", "pituitary"),
    ])
    def test_brain_mri_classes(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Histopathology ---
    @pytest.mark.parametrize("raw,expected", [
        ("colon_aca", "colon_adenocarcinoma"),
        ("colon_n", "colon_normal"),
        ("lung_aca", "lung_adenocarcinoma"),
        ("lung_n", "lung_normal"),
        ("lung_scc", "lung_squamous_cell"),
    ])
    def test_histopathology_classes(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Ultrasound ---
    @pytest.mark.parametrize("raw,expected", [
        ("benign", "benign"),
        ("malignant", "malignant"),
    ])
    def test_ultrasound_classes(self, mapper, raw, expected):
        assert mapper.normalize(raw) == expected

    # --- Unknown ---
    @pytest.mark.parametrize("raw", [
        "SomeRandomClass",
        "XYZ123",
        "completely_unknown",
    ])
    def test_unknown_returns_unknown(self, mapper, raw):
        assert mapper.normalize(raw) == "unknown"

    def test_non_string_returns_unknown(self, mapper):
        assert mapper.normalize(None) == "unknown"   # type: ignore[arg-type]

    def test_empty_string_returns_unknown(self, mapper):
        assert mapper.normalize("") == "unknown"


class TestBuildMapping:
    """Tests for build_mapping()."""

    def test_returns_dataframe(self, mapper):
        df = mapper.build_mapping(["COVID", "glioma", "benign"])
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, mapper):
        df = mapper.build_mapping(["COVID-19", "notumor"])
        assert "original_class" in df.columns
        assert "normalized_class" in df.columns

    def test_deduplicates_raw_classes(self, mapper):
        df = mapper.build_mapping(["COVID", "COVID", "glioma"])
        assert len(df) == 2  # COVID and glioma, deduplicated

    def test_mapping_df_property(self, mapper):
        assert mapper.mapping_df is None  # before build
        mapper.build_mapping(["COVID"])
        assert mapper.mapping_df is not None


class TestApply:
    """Tests for the apply() vectorised method."""

    def test_apply_returns_series(self, mapper):
        s = pd.Series(["COVID", "glioma", "benign"])
        result = mapper.apply(s)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_apply_correct_values(self, mapper):
        s = pd.Series(["COVID-19", "No Tumor", "colon_aca"])
        result = mapper.apply(s)
        assert result.tolist() == ["covid", "notumor", "colon_adenocarcinoma"]


class TestSave:
    """Tests for save()."""

    def test_save_raises_before_build(self, mapper, tmp_path):
        with pytest.raises(RuntimeError, match="build_mapping"):
            mapper.save(tmp_path)

    def test_save_creates_csv(self, mapper, tmp_path):
        mapper.build_mapping(["COVID", "glioma"])
        dest = mapper.save(tmp_path)
        assert dest.exists()
        assert dest.suffix == ".csv"

    def test_saved_csv_readable(self, mapper, tmp_path):
        mapper.build_mapping(["COVID-19", "benign"])
        dest = mapper.save(tmp_path)
        df = pd.read_csv(dest)
        assert "original_class" in df.columns
        assert "normalized_class" in df.columns

    def test_saved_csv_values(self, mapper, tmp_path):
        mapper.build_mapping(["COVID-19"])
        mapper.save(tmp_path)
        df = pd.read_csv(tmp_path / "class_mapping.csv")
        row = df[df["original_class"] == "COVID-19"]
        assert row.iloc[0]["normalized_class"] == "covid"
