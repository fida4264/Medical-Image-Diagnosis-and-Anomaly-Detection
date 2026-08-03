"""
tests/conftest.py
=================
Shared pytest fixtures and helpers for Module 1 test suite.

Key design decision: _create_dummy_image uses random pixel noise so that
the generated PNG files exceed the MIN_FILE_SIZE_BYTES (1 KB) threshold
enforced by image_utils.is_corrupted(). Solid-color PNGs compress to < 1 KB
even at large dimensions due to PNG's lossless compression.

Random pixels produce incompressible data, guaranteeing realistic file sizes.
"""

import random
import pytest
from pathlib import Path
from PIL import Image


def create_noisy_image(path: Path, size: tuple = (64, 64)) -> None:
    """
    Create a PNG image with random pixel noise at the given path.

    Using noise instead of a solid color ensures the file is NOT compressed
    away to tiny bytes by PNG's deflate codec.  This guarantees files exceed
    the project's MIN_FILE_SIZE_BYTES (1 024 bytes) threshold.

    Parameters
    ----------
    path : Path
        Destination file path (parent directories are created if missing).
    size : tuple of (int, int)
        Image dimensions (width, height).  Defaults to (64, 64) which
        produces ~12 KB PNG files — well above the 1 KB threshold.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    # Generate random RGB pixels — incompressible → large file
    pixels = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(width * height)
    ]
    img = Image.new("RGB", size)
    img.putdata(pixels)
    img.save(str(path), format="PNG")
