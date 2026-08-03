"""
tests/test_quality_report.py
==============================
Unit tests for preprocessing.reports.quality_report.QualityReportWriter.

Author: Medical AI System — Module 3
"""

import json
from pathlib import Path

import pytest

from preprocessing.reports.quality_report import QualityReportWriter


@pytest.fixture()
def writer():
    return QualityReportWriter()


class TestRecord:
    def test_record_increments_total(self, writer):
        writer.record("MRI_000001", "brain_mri", [])
        assert writer.total == 1

    def test_flagged_not_counted_for_clean_image(self, writer):
        writer.record("MRI_000001", "brain_mri", [])
        assert writer.flagged == 0

    def test_flagged_counted_for_flagged_image(self, writer):
        writer.record("MRI_000002", "brain_mri", ["blurry"])
        assert writer.flagged == 1

    def test_multiple_flags_counted_once_per_image(self, writer):
        writer.record("MRI_003", "brain_mri", ["blurry", "too_dark"])
        assert writer.flagged == 1

    def test_multiple_records(self, writer):
        for i in range(10):
            flags = ["blurry"] if i % 3 == 0 else []
            writer.record(f"MRI_{i:06d}", "brain_mri", flags)
        assert writer.total == 10


class TestSave:
    def test_save_creates_file(self, writer, tmp_path):
        writer.record("ID001", "brain_mri", [])
        path = writer.save(tmp_path / "quality_report.json")
        assert path.exists()

    def test_saved_json_valid(self, writer, tmp_path):
        writer.record("ID001", "brain_mri", ["blurry"])
        path = writer.save(tmp_path / "quality_report.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_required_top_level_keys(self, writer, tmp_path):
        writer.record("ID001", "brain_mri", [])
        path = writer.save(tmp_path / "quality_report.json")
        data = json.loads(path.read_text())
        for key in ["generated_at", "total_images", "total_flagged", "flag_counts", "per_modality", "flagged_images"]:
            assert key in data

    def test_total_images_correct(self, writer, tmp_path):
        writer.record("A", "brain_mri", [])
        writer.record("B", "chest_xray", ["blurry"])
        path = writer.save(tmp_path / "q.json")
        data = json.loads(path.read_text())
        assert data["total_images"] == 2

    def test_total_flagged_correct(self, writer, tmp_path):
        writer.record("A", "brain_mri", [])
        writer.record("B", "brain_mri", ["blurry"])
        writer.record("C", "brain_mri", ["too_dark", "blurry"])
        path = writer.save(tmp_path / "q.json")
        data = json.loads(path.read_text())
        assert data["total_flagged"] == 2

    def test_flag_counts_tracked(self, writer, tmp_path):
        writer.record("A", "brain_mri", ["blurry", "too_dark"])
        writer.record("B", "brain_mri", ["blurry"])
        path = writer.save(tmp_path / "q.json")
        data = json.loads(path.read_text())
        assert data["flag_counts"].get("blurry") == 2
        assert data["flag_counts"].get("too_dark") == 1

    def test_per_modality_structure(self, writer, tmp_path):
        writer.record("A", "brain_mri", [])
        writer.record("B", "chest_xray", ["blurry"])
        path = writer.save(tmp_path / "q.json")
        data = json.loads(path.read_text())
        assert "brain_mri" in data["per_modality"]
        assert "chest_xray" in data["per_modality"]

    def test_creates_parent_dir(self, writer, tmp_path):
        nested = tmp_path / "sub" / "quality_report.json"
        writer.record("A", "brain_mri", [])
        writer.save(nested)
        assert nested.exists()

    def test_flagged_images_list(self, writer, tmp_path):
        writer.record("MRI_001", "brain_mri", ["blurry"])
        path = writer.save(tmp_path / "q.json")
        data = json.loads(path.read_text())
        assert len(data["flagged_images"]) == 1
        assert data["flagged_images"][0]["internal_id"] == "MRI_001"

    def test_zero_total_no_division_error(self, writer, tmp_path):
        # Save with no records — should not crash
        path = writer.save(tmp_path / "empty.json")
        data = json.loads(path.read_text())
        assert data["total_images"] == 0
        assert data["flag_rate_pct"] == 0.0
