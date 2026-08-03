"""
tests/test_brain_mri_preprocessor.py
=======================================
Unit tests for BrainMRIPreprocessor.

Author: Medical AI System — Module 3
"""

import numpy as np
import pytest

from preprocessing.pipelines.brain_mri_preprocessor import BrainMRIPreprocessor

CONFIG = {
    "general": {"target_size": [224, 224]},
    "modalities": {
        "brain_mri": {
            "target_size": [224, 224],
            "output_channels": 3,
            "convert_to_grayscale": True,
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "gaussian_denoise": {"enabled": True, "kernel_size": 3, "sigma": 1.0},
            "intensity_normalize": True,
            "skull_strip_interface": False,
            "bias_field_correction": False,
        }
    },
}


@pytest.fixture()
def preprocessor():
    return BrainMRIPreprocessor(CONFIG)


def _make_mri(h=256, w=256):
    rng = np.random.RandomState(0)
    return rng.randint(10, 220, (h, w, 3), dtype=np.uint8)


class TestBrainMRIPreprocessor:

    def test_modality_property(self, preprocessor):
        assert preprocessor.modality == "brain_mri"

    def test_target_size(self, preprocessor):
        assert preprocessor.target_size == (224, 224)

    def test_output_shape(self, preprocessor):
        img = _make_mri()
        processed, _ = preprocessor.preprocess(img)
        assert processed.shape == (224, 224, 3)

    def test_output_dtype_uint8(self, preprocessor):
        processed, _ = preprocessor.preprocess(_make_mri())
        assert processed.dtype == np.uint8

    def test_operations_not_empty(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_mri())
        assert len(ops) > 0

    def test_grayscale_in_operations(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_mri())
        assert any("grayscale" in op for op in ops)

    def test_clahe_in_operations(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_mri())
        assert any("clahe" in op for op in ops)

    def test_resize_in_operations(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_mri())
        assert any("resize" in op for op in ops)

    def test_processes_different_input_sizes(self, preprocessor):
        for h, w in [(128, 128), (512, 512), (64, 300), (300, 64)]:
            img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            processed, _ = preprocessor.preprocess(img)
            assert processed.shape == (224, 224, 3), f"Failed for {h}x{w}"

    def test_processes_grayscale_input(self, preprocessor):
        gray = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        # Convert to BGR to simulate loader output
        bgr = np.stack([gray, gray, gray], axis=-1)
        processed, _ = preprocessor.preprocess(bgr)
        assert processed.shape == (224, 224, 3)

    def test_clahe_disabled(self):
        cfg = {
            "general": {"target_size": [224, 224]},
            "modalities": {
                "brain_mri": {
                    "target_size": [224, 224],
                    "clahe": {"enabled": False},
                    "gaussian_denoise": {"enabled": False},
                }
            },
        }
        p = BrainMRIPreprocessor(cfg)
        _, ops = p.preprocess(_make_mri())
        assert not any("clahe" in op for op in ops)

    def test_output_pixel_range(self, preprocessor):
        processed, _ = preprocessor.preprocess(_make_mri())
        assert processed.min() >= 0
        assert processed.max() <= 255

    def test_repr(self, preprocessor):
        r = repr(preprocessor)
        assert "brain_mri" in r
        assert "224" in r
