"""
tests/test_image_validator.py
================================
Unit tests for preprocessing.common.image_validator.ImageValidator.

Author: Medical AI System — Module 3
"""

import numpy as np
import pytest

from preprocessing.common.image_validator import (
    ImageValidator,
    FLAG_BLURRY,
    FLAG_TOO_DARK,
    FLAG_TOO_BRIGHT,
    FLAG_TOO_SMALL,
    FLAG_EMPTY,
    FLAG_BAD_CHANNELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sharp_image(h=128, w=128):
    """Create a high-frequency (sharp) image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::2, ::2] = 128
    img[1::2, 1::2] = 200
    return img


def _blurry_image(h=128, w=128):
    """Create a near-uniform (blurry) image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _dark_image(h=64, w=64):
    img = np.full((h, w, 3), 5, dtype=np.uint8)
    img[0, 0] = 6  # tiny variation to avoid zero-variance early-exit
    return img


def _bright_image(h=64, w=64):
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    img[0, 0] = 249  # tiny variation
    return img


def _empty_image(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _small_image():
    img = np.ones((16, 16, 3), dtype=np.uint8) * 100
    img[0, 0] = 101  # tiny variation
    return img


def _bad_channel_image(h=64, w=64, c=4):
    img = np.ones((h, w, c), dtype=np.uint8) * 100
    img[0, 0] = 101  # tiny variation
    return img


def _normal_image(h=128, w=128):
    """A normal, good-quality image."""
    rng = np.random.RandomState(42)
    img = rng.randint(60, 200, (h, w, 3), dtype=np.uint8)
    # Add checkerboard pattern for sharpness
    img[::4, ::4] = 255
    return img


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------

class TestIsBlurry:
    def test_blurry_image_detected(self):
        assert ImageValidator.is_blurry(_blurry_image(), threshold=80.0)

    def test_sharp_image_not_blurry(self):
        assert not ImageValidator.is_blurry(_sharp_image(), threshold=80.0)

    def test_threshold_parameter(self):
        img = _blurry_image()
        # With very low threshold, nothing is blurry
        assert not ImageValidator.is_blurry(img, threshold=0.0)


class TestIsTooDark:
    def test_dark_image_detected(self):
        assert ImageValidator.is_too_dark(_dark_image(), threshold=20.0)

    def test_normal_image_not_dark(self):
        assert not ImageValidator.is_too_dark(_normal_image(), threshold=20.0)

    def test_threshold_customization(self):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        assert ImageValidator.is_too_dark(img, threshold=60.0)
        assert not ImageValidator.is_too_dark(img, threshold=40.0)


class TestIsTooBright:
    def test_bright_image_detected(self):
        assert ImageValidator.is_too_bright(_bright_image(), threshold=245.0)

    def test_normal_image_not_bright(self):
        assert not ImageValidator.is_too_bright(_normal_image(), threshold=245.0)


class TestIsTooSmall:
    def test_small_image_detected(self):
        assert ImageValidator.is_too_small(_small_image(), min_dimension=32)

    def test_normal_size_not_small(self):
        assert not ImageValidator.is_too_small(_normal_image(), min_dimension=32)

    def test_exact_minimum_not_flagged(self):
        img = np.ones((32, 32, 3), dtype=np.uint8)
        assert not ImageValidator.is_too_small(img, min_dimension=32)


class TestIsEmpty:
    def test_empty_image_detected(self):
        assert ImageValidator.is_empty(_empty_image())

    def test_non_empty_image(self):
        assert not ImageValidator.is_empty(_normal_image())

    def test_single_color_detected_as_empty(self):
        # All pixels identical = zero variance
        uniform = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert ImageValidator.is_empty(uniform)


class TestHasBadChannels:
    def test_3_channels_ok(self):
        assert not ImageValidator.has_bad_channels(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_grayscale_ok(self):
        assert not ImageValidator.has_bad_channels(np.zeros((64, 64), dtype=np.uint8))

    def test_4_channels_bad(self):
        assert ImageValidator.has_bad_channels(np.zeros((64, 64, 4), dtype=np.uint8))

    def test_2_channels_bad(self):
        assert ImageValidator.has_bad_channels(np.zeros((64, 64, 2), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Batch validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_good_image_returns_empty_list(self):
        flags = ImageValidator.validate(_normal_image())
        assert flags == []

    def test_empty_image_returns_empty_flag(self):
        flags = ImageValidator.validate(_empty_image())
        assert FLAG_EMPTY in flags

    def test_dark_image_flagged(self):
        flags = ImageValidator.validate(_dark_image())
        assert FLAG_TOO_DARK in flags

    def test_bright_image_flagged(self):
        flags = ImageValidator.validate(_bright_image())
        assert FLAG_TOO_BRIGHT in flags

    def test_small_image_flagged(self):
        flags = ImageValidator.validate(_small_image())
        assert FLAG_TOO_SMALL in flags

    def test_bad_channels_flagged(self):
        flags = ImageValidator.validate(_bad_channel_image())
        assert FLAG_BAD_CHANNELS in flags

    def test_returns_list(self):
        assert isinstance(ImageValidator.validate(_normal_image()), list)

    def test_summary_ok(self):
        assert ImageValidator.summary([]) == "OK"

    def test_summary_flagged(self):
        s = ImageValidator.summary([FLAG_BLURRY, FLAG_TOO_DARK])
        assert "FLAGGED" in s
        assert "blurry" in s

    def test_validate_with_config(self):
        cfg = {
            "blur_threshold": 80.0,
            "dark_threshold": 20.0,
            "bright_threshold": 245.0,
            "min_dimension": 32,
        }
        flags = ImageValidator.validate_with_config(_normal_image(), cfg)
        assert isinstance(flags, list)
