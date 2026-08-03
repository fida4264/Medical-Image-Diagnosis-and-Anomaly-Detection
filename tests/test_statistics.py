"""
tests/test_statistics.py
========================
Unit tests for dataset_manager.statistics (StatisticsGenerator).

Run with:
    pytest tests/test_statistics.py -v
"""

import json
import random
import pytest
import pandas as pd
from pathlib import Path
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager.scanner import DatasetScanner
from dataset_manager.metadata_generator import MetadataGenerator
from dataset_manager.statistics import StatisticsGenerator


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


def _build_metadata_df(root: Path, outputs_dir: Path) -> pd.DataFrame:
    scanner = DatasetScanner(datasets_root=root)
    records = scanner.scan()
    gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
    return gen.generate()


@pytest.fixture
def sample_dataset(tmp_path: Path):
    root = tmp_path / "datasets"
    _create_dummy_image(root / "brain_mri" / "Glioma" / "img1.png", (128, 128))
    _create_dummy_image(root / "brain_mri" / "Normal" / "img2.png", (64, 64))
    _create_dummy_image(root / "chest_xray" / "Pneumonia" / "img3.png", (256, 256))
    outputs_dir = tmp_path / "outputs"
    df = _build_metadata_df(root, outputs_dir)
    return df, outputs_dir


# ---------------------------------------------------------------------------
# Tests — Top-Level Structure
# ---------------------------------------------------------------------------

class TestStatisticsStructure:
    def test_returns_dict(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert isinstance(stats, dict)

    def test_has_generated_at(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert "generated_at" in stats
        assert isinstance(stats["generated_at"], str)

    def test_has_overall_key(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert "overall" in stats

    def test_has_per_modality_key(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert "per_modality" in stats


# ---------------------------------------------------------------------------
# Tests — Overall Statistics
# ---------------------------------------------------------------------------

class TestOverallStatistics:
    def test_total_images_correct(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert stats["overall"]["total_images"] == 3

    def test_total_modalities_correct(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert stats["overall"]["total_modalities"] == 2

    def test_total_classes_correct(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        # Glioma, Normal (unique class names across entire dataset), Pneumonia
        assert stats["overall"]["total_classes"] == 3

    def test_file_size_bytes_positive(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert stats["overall"]["total_file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Tests — Per-Modality Statistics
# ---------------------------------------------------------------------------

class TestPerModalityStatistics:
    MODALITY_FIELDS = [
        "total_images", "valid_images", "classes", "num_classes", "splits",
        "avg_width_px", "avg_height_px", "min_width_px", "max_width_px",
        "min_height_px", "max_height_px", "avg_file_size_bytes",
        "min_file_size_bytes", "max_file_size_bytes",
        "per_class", "per_split",
    ]

    def test_brain_mri_in_per_modality(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert "brain_mri" in stats["per_modality"]

    def test_chest_xray_in_per_modality(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert "chest_xray" in stats["per_modality"]

    def test_all_modality_fields_present(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        for modality in stats["per_modality"]:
            entry = stats["per_modality"][modality]
            for field in self.MODALITY_FIELDS:
                assert field in entry, f"Field '{field}' missing in modality '{modality}'"

    def test_brain_mri_num_classes(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        assert stats["per_modality"]["brain_mri"]["num_classes"] == 2

    def test_per_class_counts_correct(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        stats = gen.compute()
        per_class = stats["per_modality"]["brain_mri"]["per_class"]
        assert per_class["Glioma"] == 1
        assert per_class["Normal"] == 1


# ---------------------------------------------------------------------------
# Tests — JSON Serialisation
# ---------------------------------------------------------------------------

class TestStatisticsJSON:
    def test_save_creates_json_file(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.compute()
        path = gen.save()
        assert path.exists()
        assert path.suffix == ".json"

    def test_saved_json_is_valid(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.compute()
        path = gen.save()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "overall" in data
        assert "per_modality" in data

    def test_save_raises_before_compute(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        with pytest.raises(RuntimeError):
            gen.save()

    def test_statistics_property_none_before_compute(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        assert gen.statistics is None

    def test_statistics_property_after_compute(self, sample_dataset) -> None:
        df, outputs_dir = sample_dataset
        gen = StatisticsGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.compute()
        assert gen.statistics is not None
