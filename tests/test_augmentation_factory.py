"""
tests/test_augmentation_factory.py
=====================================
Unit tests for augmentation.factory.AugmentationFactory.

Run with:
    pytest tests/test_augmentation_factory.py -v

Author: Medical AI System — Module 4
"""

import pytest
import yaml
from pathlib import Path

from augmentation.factory.augmentation_factory import AugmentationFactory
from augmentation.base.base_augmenter import BaseAugmenter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def factory(config):
    return AugmentationFactory(config)


# ---------------------------------------------------------------------------
# Tests — Registry
# ---------------------------------------------------------------------------

class TestFactoryRegistry:
    def test_supported_modalities_returns_list(self, factory):
        modalities = factory.supported_modalities()
        assert isinstance(modalities, list)
        assert len(modalities) == 7

    def test_all_expected_modalities_present(self, factory):
        expected = {
            "brain_mri", "brain_ct", "chest_xray",
            "retina", "skin", "ultrasound", "histopathology"
        }
        assert expected == set(factory.supported_modalities())

    def test_modalities_sorted(self, factory):
        modalities = factory.supported_modalities()
        assert modalities == sorted(modalities)

    def test_is_supported_true(self, factory):
        assert factory.is_supported("brain_mri") is True
        assert factory.is_supported("chest_xray") is True
        assert factory.is_supported("histopathology") is True

    def test_is_supported_false(self, factory):
        assert factory.is_supported("unknown_modality") is False
        assert factory.is_supported("") is False

    def test_is_supported_case_insensitive(self, factory):
        assert factory.is_supported("Brain_MRI") is True
        assert factory.is_supported("CHEST_XRAY") is True


# ---------------------------------------------------------------------------
# Tests — get()
# ---------------------------------------------------------------------------

class TestFactoryGet:
    @pytest.mark.parametrize("modality", [
        "brain_mri", "brain_ct", "chest_xray",
        "retina", "skin", "ultrasound", "histopathology"
    ])
    def test_get_returns_augmenter_instance(self, factory, modality):
        augmenter = factory.get(modality)
        assert isinstance(augmenter, BaseAugmenter)

    def test_get_raises_on_unknown_modality(self, factory):
        with pytest.raises(ValueError, match="Unknown modality"):
            factory.get("unknown_modality")

    def test_get_raises_on_empty_string(self, factory):
        with pytest.raises(ValueError):
            factory.get("")

    @pytest.mark.parametrize("modality", [
        "brain_mri", "brain_ct", "chest_xray",
        "retina", "skin", "ultrasound", "histopathology"
    ])
    def test_get_creates_new_instance_each_call(self, factory, modality):
        a1 = factory.get(modality)
        a2 = factory.get(modality)
        assert a1 is not a2

    def test_get_modality_property_correct(self, factory):
        aug = factory.get("brain_mri")
        assert aug.modality == "brain_mri"

    def test_get_strips_whitespace(self, factory):
        aug = factory.get("  brain_mri  ")
        assert aug.modality == "brain_mri"


# ---------------------------------------------------------------------------
# Tests — repr
# ---------------------------------------------------------------------------

class TestFactoryRepr:
    def test_repr_contains_factory_name(self, factory):
        r = repr(factory)
        assert "AugmentationFactory" in r

    def test_repr_contains_modality_list(self, factory):
        r = repr(factory)
        assert "brain_mri" in r
