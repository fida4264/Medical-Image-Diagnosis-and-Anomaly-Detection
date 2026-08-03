"""
tests/test_image_loader.py
============================
Unit tests for preprocessing.common.image_loader.ImageLoader.

Author: Medical AI System — Module 3
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.common.image_loader import ImageLoader, SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png(path: Path, h: int = 64, w: int = 64, channels: int = 3) -> Path:
    """Write a random PNG file."""
    img = np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_grayscale_png(path: Path, h: int = 64, w: int = 64) -> Path:
    img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def color_png(tmp_path):
    return _make_png(tmp_path / "color.png")


@pytest.fixture()
def gray_png(tmp_path):
    return _make_grayscale_png(tmp_path / "gray.png")


@pytest.fixture()
def color_jpg(tmp_path):
    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "color.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_ndarray(self, color_png):
        img = ImageLoader.load(color_png)
        assert isinstance(img, np.ndarray)

    def test_load_color_png_has_3_channels(self, color_png):
        img = ImageLoader.load(color_png)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_jpg(self, color_jpg):
        img = ImageLoader.load(color_jpg)
        assert img.shape[2] == 3

    def test_load_grayscale_as_bgr(self, gray_png):
        img = ImageLoader.load(gray_png)
        # cv2.IMREAD_COLOR forces 3 channels even for grayscale input
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_dtype_uint8(self, color_png):
        img = ImageLoader.load(color_png)
        assert img.dtype == np.uint8

    def test_load_correct_dimensions(self, tmp_path):
        path = _make_png(tmp_path / "specific.png", h=100, w=150)
        img = ImageLoader.load(path)
        assert img.shape[0] == 100
        assert img.shape[1] == 150

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageLoader.load(tmp_path / "nonexistent.png")

    def test_load_corrupt_file_raises_os_error(self, tmp_path):
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(OSError):
            ImageLoader.load(corrupt)

    def test_load_accepts_string_path(self, color_png):
        img = ImageLoader.load(str(color_png))
        assert img is not None


class TestLoadGrayscale:
    def test_load_grayscale_returns_2d(self, gray_png):
        img = ImageLoader.load_grayscale(gray_png)
        assert img.ndim == 2

    def test_load_grayscale_from_color(self, color_png):
        img = ImageLoader.load_grayscale(color_png)
        assert img.ndim == 2

    def test_load_grayscale_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageLoader.load_grayscale(tmp_path / "missing.png")


class TestInfo:
    def test_info_returns_dict(self, color_png):
        info = ImageLoader.info(color_png)
        assert isinstance(info, dict)

    def test_info_has_required_keys(self, color_png):
        info = ImageLoader.info(color_png)
        for key in ["width", "height", "mode", "format", "size_bytes"]:
            assert key in info

    def test_info_correct_dimensions(self, tmp_path):
        path = _make_png(tmp_path / "dim_test.png", h=80, w=120)
        info = ImageLoader.info(path)
        assert info["height"] == 80
        assert info["width"] == 120

    def test_info_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageLoader.info(tmp_path / "nope.png")


class TestIsSupported:
    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
    def test_supported_extensions(self, ext, tmp_path):
        fake = tmp_path / f"test{ext}"
        fake.touch()
        assert ImageLoader.is_supported(fake) is True

    def test_unsupported_extension(self, tmp_path):
        fake = tmp_path / "test.dcm"
        fake.touch()
        assert ImageLoader.is_supported(fake) is False
