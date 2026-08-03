"""
tests/test_validator.py
=======================
Unit tests for dataset_manager.validator (DatasetValidator and ValidationReport).

Run with:
    pytest tests/test_validator.py -v
"""

import json
import random
import pytest
from pathlib import Path
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager.scanner import DatasetScanner, ImageRecord
from dataset_manager.validator import DatasetValidator, ValidationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dummy_image(path: Path, size: tuple = (64, 64)) -> None:
    """Create a PNG with random pixels — guarantees file size > 1 KB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    pixels = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(w * h)]
    img = Image.new("RGB", size)
    img.putdata(pixels)
    img.save(str(path), format="PNG")


def _create_corrupted_file(path: Path) -> None:
    """Write a file with a .png extension but invalid content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\x02\x03")   # too small and not a real PNG


def _make_records_from_dir(root: Path) -> tuple:
    """Scan a tmp datasets directory and return (records, discovered_modalities)."""
    scanner = DatasetScanner(datasets_root=root)
    records = scanner.scan()
    return records, scanner.discovered_modalities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    # Use distinct sizes so the images have different MD5 hashes (no false duplicates)
    _create_dummy_image(root / "brain_mri" / "Glioma" / "img1.png", (256, 256))
    _create_dummy_image(root / "brain_mri" / "Normal" / "img2.png", (256, 200))
    return root


@pytest.fixture
def corrupted_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    _create_dummy_image(root / "brain_mri" / "Glioma" / "good.png", (256, 256))
    _create_corrupted_file(root / "brain_mri" / "Glioma" / "bad.png")
    return root


@pytest.fixture
def duplicate_dataset(tmp_path: Path) -> Path:
    """Two files with identical content (different names)."""
    root = tmp_path / "datasets"
    _create_dummy_image(root / "brain_mri" / "Glioma" / "img1.png", (256, 256))
    import shutil
    src = root / "brain_mri" / "Glioma" / "img1.png"
    dst = root / "brain_mri" / "Glioma" / "img2.png"
    shutil.copy(str(src), str(dst))
    return root


# ---------------------------------------------------------------------------
# Tests — Clean Dataset (no issues expected)
# ---------------------------------------------------------------------------

class TestCleanDataset:
    def test_no_errors_on_clean_dataset(self, clean_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(clean_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=clean_dataset.parent / "outputs",
        )
        report = validator.validate()
        assert report.errors == 0

    def test_report_images_scanned(self, clean_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(clean_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=clean_dataset.parent / "outputs",
        )
        report = validator.validate()
        assert report.total_images_scanned == 2


# ---------------------------------------------------------------------------
# Tests — Corrupted Images
# ---------------------------------------------------------------------------

class TestCorruptedImages:
    def test_detects_corrupted_image(self, corrupted_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(corrupted_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=corrupted_dataset.parent / "outputs",
        )
        report = validator.validate()
        corrupted_issues = [
            i for i in report.issues if i.category == "CORRUPTED"
        ]
        assert len(corrupted_issues) >= 1

    def test_corrupted_issues_are_errors(self, corrupted_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(corrupted_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=corrupted_dataset.parent / "outputs",
        )
        report = validator.validate()
        for issue in report.issues:
            if issue.category == "CORRUPTED":
                assert issue.severity == "ERROR"


# ---------------------------------------------------------------------------
# Tests — Duplicate Detection
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_detects_duplicates(self, duplicate_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(duplicate_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=duplicate_dataset.parent / "outputs",
        )
        report = validator.validate()
        dup_issues = [i for i in report.issues if i.category == "DUPLICATE"]
        assert len(dup_issues) >= 1

    def test_duplicate_group_has_multiple_paths(self, duplicate_dataset: Path) -> None:
        records, modalities = _make_records_from_dir(duplicate_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=duplicate_dataset.parent / "outputs",
        )
        report = validator.validate()
        for paths in report.duplicate_groups.values():
            assert len(paths) >= 2


# ---------------------------------------------------------------------------
# Tests — Validation Report Serialisation
# ---------------------------------------------------------------------------

class TestValidationReportSave:
    def test_save_creates_json_file(self, clean_dataset: Path) -> None:
        outputs_dir = clean_dataset.parent / "outputs"
        records, modalities = _make_records_from_dir(clean_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=outputs_dir,
        )
        validator.validate()
        report_path = validator.save_report()
        assert report_path.exists()
        assert report_path.suffix == ".json"

    def test_saved_json_is_valid(self, clean_dataset: Path) -> None:
        outputs_dir = clean_dataset.parent / "outputs"
        records, modalities = _make_records_from_dir(clean_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=outputs_dir,
        )
        validator.validate()
        report_path = validator.save_report()
        with open(report_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "total_images_scanned" in data
        assert "issues" in data
        assert "errors" in data
        assert "warnings" in data

    def test_saved_json_contains_issues_list(self, corrupted_dataset: Path) -> None:
        outputs_dir = corrupted_dataset.parent / "outputs"
        records, modalities = _make_records_from_dir(corrupted_dataset)
        validator = DatasetValidator(
            records=records,
            discovered_modalities=modalities,
            outputs_dir=outputs_dir,
        )
        validator.validate()
        report_path = validator.save_report()
        with open(report_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data["issues"], list)
        assert len(data["issues"]) > 0


# ---------------------------------------------------------------------------
# Tests — ValidationReport model
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_add_error_increments_counter(self) -> None:
        from dataset_manager.validator import ValidationIssue
        report = ValidationReport()
        report.add_issue(ValidationIssue(
            severity="ERROR",
            category="CORRUPTED",
            description="test",
            affected_path="/tmp/x.png",
        ))
        assert report.errors == 1
        assert report.warnings == 0
        assert report.total_issues == 1

    def test_add_warning_increments_counter(self) -> None:
        from dataset_manager.validator import ValidationIssue
        report = ValidationReport()
        report.add_issue(ValidationIssue(
            severity="WARNING",
            category="DUPLICATE",
            description="test",
            affected_path="/tmp/x.png",
        ))
        assert report.warnings == 1
        assert report.errors == 0

    def test_to_dict_is_json_serialisable(self) -> None:
        report = ValidationReport(total_images_scanned=100)
        d = report.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
