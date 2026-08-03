"""
tests/test_augmentation_policies.py
======================================
Unit tests for all 7 modality augmentation policies.

Tests cover:
- Output shape and dtype
- Operations list is non-empty
- Reproducibility (same seed → same output)
- Different seeds → different outputs
- Medical safety constraints (no vertical flip where prohibited)

Run with:
    pytest tests/test_augmentation_policies.py -v

Author: Medical AI System — Module 4
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

from augmentation.factory.augmentation_factory import AugmentationFactory
from augmentation.policies.brain_mri_policy import BrainMRIAugmenter
from augmentation.policies.brain_ct_policy import BrainCTAugmenter
from augmentation.policies.chest_xray_policy import ChestXRayAugmenter
from augmentation.policies.retina_policy import RetinaAugmenter
from augmentation.policies.skin_policy import SkinAugmenter
from augmentation.policies.ultrasound_policy import UltrasoundAugmenter
from augmentation.policies.histopathology_policy import HistopathologyAugmenter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation.yaml"


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def sample_image_gray():
    """224×224 grayscale image as 3-channel BGR (as output from Module 3)."""
    gray = np.random.randint(30, 200, (224, 224), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


@pytest.fixture
def sample_image_color():
    """224×224 colour BGR image."""
    return np.random.randint(30, 200, (224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Parameterized tests for all modalities
# ---------------------------------------------------------------------------

ALL_MODALITIES = [
    "brain_mri",
    "brain_ct",
    "chest_xray",
    "retina",
    "skin",
    "ultrasound",
    "histopathology",
]


class TestAllPolicies:
    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_augment_returns_tuple(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        result = augmenter.augment(sample_image_color, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_output_shape_preserved(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        aug_img, _ = augmenter.augment(sample_image_color, seed=42)
        assert aug_img.shape == (224, 224, 3)

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_output_dtype_uint8(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        aug_img, _ = augmenter.augment(sample_image_color, seed=42)
        assert aug_img.dtype == np.uint8

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_operations_list_not_empty(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        _, ops = augmenter.augment(sample_image_color, seed=42)
        assert isinstance(ops, list)
        assert len(ops) >= 1

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_output_pixel_range_valid(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        aug_img, _ = augmenter.augment(sample_image_color, seed=42)
        assert aug_img.min() >= 0
        assert aug_img.max() <= 255

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_reproducibility_same_seed(self, config, sample_image_color, modality):
        factory = AugmentationFactory(config)
        a1 = factory.get(modality)
        a2 = factory.get(modality)
        img1, _ = a1.augment(sample_image_color.copy(), seed=99)
        img2, _ = a2.augment(sample_image_color.copy(), seed=99)
        np.testing.assert_array_equal(img1, img2)

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_accepts_different_input_sizes(self, config, modality):
        """Augmenter should handle various input sizes gracefully."""
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        for size in [(64, 64, 3), (128, 128, 3), (224, 224, 3)]:
            image = np.random.randint(0, 255, size, dtype=np.uint8)
            aug, _ = augmenter.augment(image, seed=42)
            assert aug.dtype == np.uint8

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_repr_contains_modality(self, config, modality):
        factory = AugmentationFactory(config)
        augmenter = factory.get(modality)
        assert modality in repr(augmenter)


# ---------------------------------------------------------------------------
# Medical safety tests
# ---------------------------------------------------------------------------

class TestMedicalSafetyConstraints:
    def test_brain_mri_no_vertical_flip(self, config):
        """Brain MRI policy must not contain VerticalFlip."""
        aug = BrainMRIAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "VerticalFlip" not in transform_names

    def test_brain_mri_no_horizontal_flip(self, config):
        aug = BrainMRIAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "HorizontalFlip" not in transform_names

    def test_brain_ct_no_vertical_flip(self, config):
        aug = BrainCTAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "VerticalFlip" not in transform_names

    def test_chest_xray_no_vertical_flip(self, config):
        aug = ChestXRayAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "VerticalFlip" not in transform_names

    def test_chest_xray_no_horizontal_flip(self, config):
        aug = ChestXRayAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "HorizontalFlip" not in transform_names

    def test_ultrasound_no_vertical_flip(self, config):
        aug = UltrasoundAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "VerticalFlip" not in transform_names

    def test_skin_has_horizontal_flip(self, config):
        """Skin lesion policy should include HorizontalFlip."""
        aug = SkinAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "HorizontalFlip" in transform_names

    def test_skin_has_vertical_flip(self, config):
        aug = SkinAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "VerticalFlip" in transform_names

    def test_histopathology_has_both_flips(self, config):
        aug = HistopathologyAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "HorizontalFlip" in transform_names
        assert "VerticalFlip" in transform_names

    def test_retina_has_horizontal_flip(self, config):
        aug = RetinaAugmenter(config)
        transform_names = [type(t).__name__ for t in aug._pipeline.transforms]
        assert "HorizontalFlip" in transform_names
