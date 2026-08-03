"""
tests/test_preprocessor_factory.py
=====================================
Unit tests for preprocessing.factory.preprocessor_factory.PreprocessorFactory.

Author: Medical AI System — Module 3
"""

import numpy as np
import pytest

from preprocessing.factory.preprocessor_factory import PreprocessorFactory
from preprocessing.base.base_preprocessor import BasePreprocessor

MINIMAL_CONFIG = {
    "general": {"target_size": [224, 224], "output_format": "png"},
    "modalities": {
        "brain_mri": {"target_size": [224, 224], "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]}, "gaussian_denoise": {"enabled": True}},
        "brain_ct":  {"target_size": [224, 224], "windowing": {"enabled": True, "window_level": 40, "window_width": 80}, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]}},
        "chest_xray": {"target_size": [224, 224], "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": [8, 8]}},
        "retina": {"target_size": [224, 224], "circular_crop": {"enabled": True}, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]}},
        "skin": {"target_size": [224, 224], "color_normalize": {"enabled": True}, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]}},
        "ultrasound": {"target_size": [224, 224], "speckle_reduction": {"enabled": True, "method": "median", "kernel_size": 3}, "clahe": {"enabled": True, "clip_limit": 2.5, "tile_grid_size": [8, 8]}},
        "histopathology": {"target_size": [224, 224], "color_normalize": {"enabled": True}},
    },
    "quality": {"blur_threshold": 80.0, "dark_threshold": 20.0, "bright_threshold": 245.0, "min_dimension": 32},
}

ALL_MODALITIES = [
    "brain_mri", "brain_ct", "chest_xray",
    "retina", "skin", "ultrasound", "histopathology",
]


@pytest.fixture()
def factory():
    return PreprocessorFactory(MINIMAL_CONFIG)


class TestGetPreprocessor:
    """Tests for factory.get()."""

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_get_returns_base_preprocessor(self, factory, modality):
        preprocessor = factory.get(modality)
        assert isinstance(preprocessor, BasePreprocessor)

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_get_correct_modality_property(self, factory, modality):
        preprocessor = factory.get(modality)
        assert preprocessor.modality == modality

    def test_get_unknown_modality_raises(self, factory):
        with pytest.raises(ValueError, match="Unknown modality"):
            factory.get("xray_unknown")

    def test_get_empty_string_raises(self, factory):
        with pytest.raises(ValueError):
            factory.get("")

    def test_get_case_insensitive(self, factory):
        # Normalised to lowercase before lookup
        preprocessor = factory.get("brain_mri")
        assert preprocessor is not None

    def test_returns_fresh_instance_each_call(self, factory):
        p1 = factory.get("brain_mri")
        p2 = factory.get("brain_mri")
        assert p1 is not p2  # separate instances


class TestSupportedModalities:
    def test_returns_list(self):
        result = PreprocessorFactory.supported_modalities()
        assert isinstance(result, list)

    def test_all_7_modalities_present(self):
        supported = PreprocessorFactory.supported_modalities()
        for mod in ALL_MODALITIES:
            assert mod in supported

    def test_returns_sorted(self):
        supported = PreprocessorFactory.supported_modalities()
        assert supported == sorted(supported)


class TestIsSupported:
    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_known_modality_returns_true(self, modality):
        assert PreprocessorFactory.is_supported(modality) is True

    def test_unknown_modality_returns_false(self):
        assert PreprocessorFactory.is_supported("xray_unknown") is False


class TestPreprocess:
    """Smoke test: factory.get(modality).preprocess(image) returns correct shape."""

    def _make_image(self, h=128, w=128, channels=3):
        rng = np.random.RandomState(42)
        return rng.randint(30, 220, (h, w, channels), dtype=np.uint8)

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_preprocess_returns_correct_shape(self, factory, modality):
        img = self._make_image()
        preprocessor = factory.get(modality)
        processed, ops = preprocessor.preprocess(img)
        assert processed.shape == (224, 224, 3)

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_preprocess_returns_uint8(self, factory, modality):
        img = self._make_image()
        preprocessor = factory.get(modality)
        processed, _ = preprocessor.preprocess(img)
        assert processed.dtype == np.uint8

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_preprocess_returns_operations_list(self, factory, modality):
        img = self._make_image()
        preprocessor = factory.get(modality)
        _, ops = preprocessor.preprocess(img)
        assert isinstance(ops, list)
        assert len(ops) > 0

    @pytest.mark.parametrize("modality", ALL_MODALITIES)
    def test_operations_are_strings(self, factory, modality):
        img = self._make_image()
        preprocessor = factory.get(modality)
        _, ops = preprocessor.preprocess(img)
        assert all(isinstance(op, str) for op in ops)
