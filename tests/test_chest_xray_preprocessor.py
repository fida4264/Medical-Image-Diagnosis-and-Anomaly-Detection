"""
tests/test_chest_xray_preprocessor.py
========================================
Unit tests for ChestXRayPreprocessor.

Author: Medical AI System — Module 3
"""

import numpy as np
import pytest

from preprocessing.pipelines.chest_xray_preprocessor import ChestXRayPreprocessor

CONFIG = {
    "general": {"target_size": [224, 224]},
    "modalities": {
        "chest_xray": {
            "target_size": [224, 224],
            "output_channels": 3,
            "convert_to_grayscale": True,
            "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid_size": [8, 8]},
            "histogram_equalization": False,
            "noise_reduction": {"enabled": True, "method": "gaussian", "kernel_size": 3},
            "intensity_normalize": True,
        }
    },
}


@pytest.fixture()
def preprocessor():
    return ChestXRayPreprocessor(CONFIG)


def _make_xray(h=256, w=256):
    rng = np.random.RandomState(7)
    return rng.randint(20, 240, (h, w, 3), dtype=np.uint8)


class TestChestXRayPreprocessor:

    def test_modality_property(self, preprocessor):
        assert preprocessor.modality == "chest_xray"

    def test_output_shape(self, preprocessor):
        processed, _ = preprocessor.preprocess(_make_xray())
        assert processed.shape == (224, 224, 3)

    def test_output_dtype(self, preprocessor):
        processed, _ = preprocessor.preprocess(_make_xray())
        assert processed.dtype == np.uint8

    def test_operations_list_not_empty(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_xray())
        assert len(ops) > 0

    def test_grayscale_conversion_in_ops(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_xray())
        assert any("grayscale" in op for op in ops)

    def test_clahe_in_ops(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_xray())
        assert any("clahe" in op for op in ops)

    def test_resize_in_ops(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_xray())
        assert any("resize" in op for op in ops)

    def test_histogram_eq_when_enabled(self):
        cfg = {
            "general": {"target_size": [224, 224]},
            "modalities": {
                "chest_xray": {
                    "target_size": [224, 224],
                    "clahe": {"enabled": False},
                    "histogram_equalization": True,
                    "noise_reduction": {"enabled": False},
                }
            },
        }
        p = ChestXRayPreprocessor(cfg)
        _, ops = p.preprocess(_make_xray())
        assert any("histogram_eq" in op for op in ops)

    def test_histogram_eq_when_disabled(self, preprocessor):
        _, ops = preprocessor.preprocess(_make_xray())
        assert not any("histogram_eq" in op for op in ops)

    def test_pixel_range_valid(self, preprocessor):
        processed, _ = preprocessor.preprocess(_make_xray())
        assert processed.min() >= 0
        assert processed.max() <= 255

    def test_various_input_shapes(self, preprocessor):
        for h, w in [(128, 128), (512, 256), (100, 400)]:
            img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            processed, _ = preprocessor.preprocess(img)
            assert processed.shape == (224, 224, 3)

    def test_3ch_output_from_grayscale_source(self, preprocessor):
        # Simulate true grayscale input (all channels equal)
        gray_vals = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        img = np.stack([gray_vals, gray_vals, gray_vals], axis=-1)
        processed, _ = preprocessor.preprocess(img)
        assert processed.shape[2] == 3
