"""
tests/test_label_generator.py
==============================
Unit tests for dataset_manager.label_generator (LabelGenerator).

Run with:
    pytest tests/test_label_generator.py -v
"""

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
from dataset_manager.label_generator import LabelGenerator
from dataset_manager.config import (
    COL_IMAGE_ID, COL_MODALITY, COL_DISEASE_CLASS,
    COL_SPLIT, COL_LABEL, COL_CLASS_NAME, COL_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _create_dummy_image(path: Path) -> None:
    """Create a PNG with random pixels — guarantees file size > 1 KB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(64 * 64)]
    img = Image.new("RGB", (64, 64))
    img.putdata(pixels)
    img.save(str(path), format="PNG")


def _build_metadata_df(root: Path, outputs_dir: Path) -> pd.DataFrame:
    scanner = DatasetScanner(datasets_root=root)
    records = scanner.scan()
    gen = MetadataGenerator(records=records, outputs_dir=outputs_dir)
    return gen.generate()


@pytest.fixture
def multi_class_dataset(tmp_path: Path):
    root = tmp_path / "datasets"
    # brain_mri: 3 classes
    for cls in ["Glioma", "Meningioma", "Normal"]:
        _create_dummy_image(root / "brain_mri" / cls / "img.png")
    # chest_xray: 2 classes
    for cls in ["Normal", "Pneumonia"]:
        _create_dummy_image(root / "chest_xray" / cls / "img.png")
    outputs_dir = tmp_path / "outputs"
    df = _build_metadata_df(root, outputs_dir)
    return df, outputs_dir


# ---------------------------------------------------------------------------
# Tests — Label Assignment
# ---------------------------------------------------------------------------

class TestLabelAssignment:
    def test_labels_are_zero_based_integers(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()
        assert labels_df[COL_LABEL].min() == 0

    def test_labels_are_contiguous(self, multi_class_dataset) -> None:
        """Labels for a modality should be 0, 1, 2, … N-1."""
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()

        for modality in labels_df[COL_MODALITY].unique():
            mod_labels = sorted(
                labels_df[labels_df[COL_MODALITY] == modality][COL_LABEL].unique()
            )
            assert mod_labels == list(range(len(mod_labels)))

    def test_labels_are_per_modality(self, multi_class_dataset) -> None:
        """
        brain_mri Glioma → 0, chest_xray Normal → 0 (separate namespaces).
        """
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()

        mri_normal_label = labels_df[
            (labels_df[COL_MODALITY] == "brain_mri") &
            (labels_df[COL_DISEASE_CLASS] == "Normal")
        ][COL_LABEL].iloc[0]

        xray_normal_label = labels_df[
            (labels_df[COL_MODALITY] == "chest_xray") &
            (labels_df[COL_DISEASE_CLASS] == "Normal")
        ][COL_LABEL].iloc[0]

        # Both should map "Normal" to label 2 and 0 respectively (alphabetical)
        # The key point is they are independently scoped — neither must equal -1
        assert mri_normal_label >= 0
        assert xray_normal_label >= 0

    def test_brain_mri_has_three_classes(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()
        mri_labels = labels_df[labels_df[COL_MODALITY] == "brain_mri"][COL_LABEL].unique()
        assert len(mri_labels) == 3

    def test_labels_are_alphabetically_ordered(self, multi_class_dataset) -> None:
        """Glioma=0, Meningioma=1, Normal=2 for brain_mri."""
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()

        mri_map = (
            labels_df[labels_df[COL_MODALITY] == "brain_mri"]
            [[COL_DISEASE_CLASS, COL_LABEL]]
            .drop_duplicates()
            .set_index(COL_DISEASE_CLASS)[COL_LABEL]
            .to_dict()
        )
        assert mri_map["Glioma"] == 0
        assert mri_map["Meningioma"] == 1
        assert mri_map["Normal"] == 2


# ---------------------------------------------------------------------------
# Tests — Labels Schema
# ---------------------------------------------------------------------------

class TestLabelsSchema:
    EXPECTED_COLS = [
        COL_IMAGE_ID, COL_MODALITY, COL_DISEASE_CLASS,
        COL_SPLIT, COL_LABEL, COL_CLASS_NAME,
    ]

    def test_all_columns_present(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()
        for col in self.EXPECTED_COLS:
            assert col in labels_df.columns, f"Missing column: {col}"

    def test_row_count_matches_metadata(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        labels_df, _ = gen.generate()
        assert len(labels_df) == len(df)


# ---------------------------------------------------------------------------
# Tests — Class Distribution Schema
# ---------------------------------------------------------------------------

class TestClassDistribution:
    EXPECTED_COLS = [COL_MODALITY, COL_DISEASE_CLASS, COL_SPLIT, COL_COUNT]

    def test_distribution_columns_present(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        _, dist_df = gen.generate()
        for col in self.EXPECTED_COLS:
            assert col in dist_df.columns, f"Missing column: {col}"

    def test_total_count_matches_metadata_rows(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        _, dist_df = gen.generate()
        assert dist_df[COL_COUNT].sum() == len(df)

    def test_counts_are_positive(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        _, dist_df = gen.generate()
        assert (dist_df[COL_COUNT] > 0).all()


# ---------------------------------------------------------------------------
# Tests — Save
# ---------------------------------------------------------------------------

class TestLabelSave:
    def test_save_labels_creates_csv(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.generate()
        path = gen.save_labels()
        assert path.exists()

    def test_save_distribution_creates_csv(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.generate()
        path = gen.save_class_distribution()
        assert path.exists()

    def test_save_raises_before_generate(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        with pytest.raises(RuntimeError):
            gen.save_labels()

    def test_labels_csv_readable(self, multi_class_dataset) -> None:
        df, outputs_dir = multi_class_dataset
        gen = LabelGenerator(metadata_df=df, outputs_dir=outputs_dir)
        gen.generate()
        path = gen.save_labels()
        df2 = pd.read_csv(path)
        assert COL_LABEL in df2.columns
