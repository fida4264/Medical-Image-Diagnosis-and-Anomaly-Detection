"""
tests/test_scanner.py
=====================
Unit tests for dataset_manager.scanner (DatasetScanner and ImageRecord).

Uses temporary directories created via pytest's `tmp_path` fixture so that
tests are hermetic and do not depend on the presence of real dataset files.

Run with:
    pytest tests/test_scanner.py -v
"""

import random
import pytest
from pathlib import Path
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager.scanner import DatasetScanner, ImageRecord


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_dataset(tmp_path: Path) -> Path:
    """
    Layout A — flat disease folders:
        datasets/
            brain_mri/
                Glioma/
                    img1.png
                Normal/
                    img2.png
    """
    root = tmp_path / "datasets"
    _create_dummy_image(root / "brain_mri" / "Glioma" / "img1.png")
    _create_dummy_image(root / "brain_mri" / "Normal" / "img2.png")
    return root


@pytest.fixture
def split_dataset(tmp_path: Path) -> Path:
    """
    Layout B — split sub-folders inside each disease folder:
        datasets/
            chest_xray/
                Pneumonia/
                    train/
                        img1.png
                    test/
                        img2.png
                Normal/
                    train/
                        img3.png
    """
    root = tmp_path / "datasets"
    _create_dummy_image(root / "chest_xray" / "Pneumonia" / "train" / "img1.png")
    _create_dummy_image(root / "chest_xray" / "Pneumonia" / "test" / "img2.png")
    _create_dummy_image(root / "chest_xray" / "Normal" / "train" / "img3.png")
    return root


@pytest.fixture
def split_first_dataset(tmp_path: Path) -> Path:
    """
    Layout C — split folders first, then disease:
        datasets/
            skin/
                train/
                    Melanoma/
                        img1.png
                val/
                    Melanoma/
                        img2.png
    """
    root = tmp_path / "datasets"
    _create_dummy_image(root / "skin" / "train" / "Melanoma" / "img1.png")
    _create_dummy_image(root / "skin" / "val" / "Melanoma" / "img2.png")
    return root


@pytest.fixture
def mixed_dataset(tmp_path: Path) -> Path:
    """Dataset containing multiple modalities in mixed layouts."""
    root = tmp_path / "datasets"
    # Layout A modality
    _create_dummy_image(root / "retina" / "Diabetic" / "img1.png")
    # Layout B modality
    _create_dummy_image(root / "brain_ct" / "Hemorrhage" / "train" / "img2.png")
    return root


# ---------------------------------------------------------------------------
# Tests — DatasetScanner existence checks
# ---------------------------------------------------------------------------

class TestDatasetScannerInit:
    def test_raises_when_root_missing(self, tmp_path: Path) -> None:
        """Scanner must raise FileNotFoundError when root doesn't exist."""
        scanner = DatasetScanner(datasets_root=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="not found"):
            scanner.scan()

    def test_scan_empty_root(self, tmp_path: Path) -> None:
        """Scanner on an empty datasets root should return an empty list."""
        root = tmp_path / "datasets"
        root.mkdir()
        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        assert isinstance(records, list)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Tests — Layout A (flat disease folders)
# ---------------------------------------------------------------------------

class TestLayoutA:
    def test_discovers_correct_count(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        assert len(records) == 2

    def test_modality_detected(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        scanner.scan()
        assert "brain_mri" in scanner.discovered_modalities

    def test_disease_classes_detected(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        classes = {r.disease_class for r in records}
        assert "Glioma" in classes
        assert "Normal" in classes

    def test_split_is_unspecified(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        assert all(r.split == "unspecified" for r in records)

    def test_image_record_is_frozen(self, flat_dataset: Path) -> None:
        """ImageRecord must be immutable (frozen dataclass)."""
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        with pytest.raises((AttributeError, TypeError)):
            records[0].modality = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — Layout B (disease → split)
# ---------------------------------------------------------------------------

class TestLayoutB:
    def test_discovers_correct_count(self, split_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_dataset)
        records = scanner.scan()
        assert len(records) == 3

    def test_splits_detected(self, split_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_dataset)
        records = scanner.scan()
        splits = {r.split for r in records}
        assert "train" in splits
        assert "test" in splits

    def test_no_unspecified_split(self, split_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_dataset)
        records = scanner.scan()
        assert all(r.split != "unspecified" for r in records)


# ---------------------------------------------------------------------------
# Tests — Layout C (split → disease)
# ---------------------------------------------------------------------------

class TestLayoutC:
    def test_discovers_correct_count(self, split_first_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_first_dataset)
        records = scanner.scan()
        assert len(records) == 2

    def test_splits_detected(self, split_first_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_first_dataset)
        records = scanner.scan()
        splits = {r.split for r in records}
        assert "train" in splits
        assert "val" in splits

    def test_disease_class_detected(self, split_first_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=split_first_dataset)
        records = scanner.scan()
        assert all(r.disease_class == "Melanoma" for r in records)


# ---------------------------------------------------------------------------
# Tests — Hidden and Unsupported Files
# ---------------------------------------------------------------------------

class TestFileFiltering:
    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        root = tmp_path / "datasets"
        _create_dummy_image(root / "skin" / "Benign" / "img1.png")
        hidden = root / "skin" / "Benign" / ".hidden.png"
        _create_dummy_image(hidden)

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        paths = [r.image_path for r in records]
        assert not any(".hidden" in str(p) for p in paths)
        assert scanner.skipped_hidden > 0

    def test_skips_unsupported_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "datasets"
        _create_dummy_image(root / "retina" / "Normal" / "img1.png")
        txt_file = root / "retina" / "Normal" / "readme.txt"
        txt_file.parent.mkdir(parents=True, exist_ok=True)
        txt_file.write_text("not an image")

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        assert all(r.filename.endswith(".png") for r in records)
        assert scanner.skipped_unsupported > 0

    def test_supports_multiple_extensions(self, tmp_path: Path) -> None:
        root = tmp_path / "datasets"
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            p = root / "skin" / "Benign" / f"image{ext}"
            _create_dummy_image(p)

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        exts = {r.extension for r in records}
        assert ".jpg" in exts
        assert ".png" in exts
        assert ".bmp" in exts


# ---------------------------------------------------------------------------
# Tests — ImageRecord attributes
# ---------------------------------------------------------------------------

class TestImageRecord:
    def test_image_id_uses_forward_slashes(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        for rec in records:
            assert "\\" not in rec.image_id, (
                f"image_id should use forward slashes, got: {rec.image_id}"
            )

    def test_extension_is_lower_case(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        for rec in records:
            assert rec.extension == rec.extension.lower()

    def test_file_size_positive(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        for rec in records:
            assert rec.file_size_bytes > 0

    def test_image_path_is_absolute(self, flat_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=flat_dataset)
        records = scanner.scan()
        for rec in records:
            assert rec.image_path.is_absolute()


# ---------------------------------------------------------------------------
# Tests — Multiple Modalities
# ---------------------------------------------------------------------------

class TestMultipleModalities:
    def test_mixed_layout_correct_count(self, mixed_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=mixed_dataset)
        records = scanner.scan()
        assert len(records) == 2

    def test_modality_count(self, mixed_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=mixed_dataset)
        scanner.scan()
        assert scanner.modality_count == 2

    def test_each_record_has_correct_modality(self, mixed_dataset: Path) -> None:
        scanner = DatasetScanner(datasets_root=mixed_dataset)
        records = scanner.scan()
        modalities = {r.modality for r in records}
        assert modalities == {"retina", "brain_ct"}
