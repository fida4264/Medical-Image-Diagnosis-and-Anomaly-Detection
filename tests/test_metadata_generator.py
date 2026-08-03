"""
tests/test_metadata_generator.py
=================================
Unit tests for dataset_manager.metadata_generator (MetadataGenerator).

Run with:
    pytest tests/test_metadata_generator.py -v
"""

import random
import pytest
from pathlib import Path
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager.scanner import DatasetScanner
from dataset_manager.metadata_generator import MetadataGenerator
from dataset_manager.config import (
    COL_IMAGE_ID, COL_MODALITY, COL_DISEASE_CLASS, COL_SPLIT,
    COL_IMAGE_PATH, COL_FILENAME, COL_EXTENSION,
    COL_WIDTH, COL_HEIGHT, COL_FILE_SIZE_BYTES, COL_IS_VALID, COL_MD5_HASH,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
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


@pytest.fixture
def simple_dataset(tmp_path: Path):
    root = tmp_path / "datasets"
    # Use distinct sizes to ensure different MD5 hashes and > 1 KB files
    _create_dummy_image(root / "brain_mri" / "Glioma" / "img1.png", (256, 256))
    _create_dummy_image(root / "brain_mri" / "Normal" / "img2.png", (256, 200))
    scanner = DatasetScanner(datasets_root=root)
    records = scanner.scan()
    return records, tmp_path / "outputs"


# ---------------------------------------------------------------------------
# Tests — Schema Validation
# ---------------------------------------------------------------------------

class TestMetadataSchema:
    EXPECTED_COLUMNS = [
        COL_IMAGE_ID, COL_MODALITY, COL_DISEASE_CLASS, COL_SPLIT,
        COL_IMAGE_PATH, COL_FILENAME, COL_EXTENSION,
        COL_WIDTH, COL_HEIGHT, COL_FILE_SIZE_BYTES, COL_IS_VALID, COL_MD5_HASH,
    ]

    def test_all_expected_columns_present(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        for col in self.EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_records(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        assert len(df) == len(records)

    def test_no_null_image_ids(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        assert df[COL_IMAGE_ID].notna().all()

    def test_no_null_modalities(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        assert df[COL_MODALITY].notna().all()


# ---------------------------------------------------------------------------
# Tests — Dimension Accuracy
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_width_height_populated(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        # All images are valid so dimensions should be > 0
        assert (df[COL_WIDTH] > 0).all()
        assert (df[COL_HEIGHT] > 0).all()

    def test_dimensions_match_actual_image(self, tmp_path: Path) -> None:
        root = tmp_path / "datasets"
        target_w, target_h = 200, 150
        _create_dummy_image(root / "skin" / "Benign" / "img.png", (target_w, target_h))

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        gen = MetadataGenerator(records=records, outputs_dir=tmp_path / "outputs")
        df = gen.generate()

        row = df[df[COL_FILENAME] == "img.png"].iloc[0]
        assert int(row[COL_WIDTH]) == target_w
        assert int(row[COL_HEIGHT]) == target_h


# ---------------------------------------------------------------------------
# Tests — MD5 and Validity
# ---------------------------------------------------------------------------

class TestMD5AndValidity:
    def test_md5_hash_is_32_chars(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        valid_hashes = df[df[COL_IS_VALID]][COL_MD5_HASH]
        assert (valid_hashes.str.len() == 32).all()

    def test_is_valid_true_for_good_images(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        df = gen.generate()
        assert df[COL_IS_VALID].all()

    def test_md5_unique_per_unique_image(self, tmp_path: Path) -> None:
        """Different images should (almost certainly) have different MD5 hashes."""
        root = tmp_path / "datasets"
        _create_dummy_image(root / "retina" / "Healthy" / "img1.png", (32, 32))
        _create_dummy_image(root / "retina" / "Healthy" / "img2.png", (64, 64))

        scanner = DatasetScanner(datasets_root=root)
        records = scanner.scan()
        gen = MetadataGenerator(records=records, outputs_dir=tmp_path / "outputs")
        df = gen.generate()
        assert df[COL_MD5_HASH].nunique() == 2


# ---------------------------------------------------------------------------
# Tests — Save
# ---------------------------------------------------------------------------

class TestMetadataSave:
    def test_save_creates_csv(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        gen.generate()
        path = gen.save()
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_raises_before_generate(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        with pytest.raises(RuntimeError):
            gen.save()

    def test_csv_can_be_read_back(self, simple_dataset) -> None:
        import pandas as pd
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        gen.generate()
        path = gen.save()
        df2 = pd.read_csv(path)
        assert len(df2) == len(records)
        assert COL_IMAGE_ID in df2.columns

    def test_dataframe_property_is_none_before_generate(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        assert gen.dataframe is None

    def test_dataframe_property_after_generate(self, simple_dataset) -> None:
        records, outputs_dir = simple_dataset
        gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
        gen.generate()
        assert gen.dataframe is not None
