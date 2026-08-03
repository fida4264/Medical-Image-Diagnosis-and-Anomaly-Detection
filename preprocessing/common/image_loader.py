"""
preprocessing/common/image_loader.py
=======================================
Robust image loader with OpenCV primary and Pillow fallback.

Always returns a BGR uint8 numpy array regardless of the source format.
Supports: JPEG, PNG, BMP, TIFF, WEBP, and any format Pillow can open.

Classes
-------
ImageLoader
    Static/class methods for loading images from file paths.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}


class ImageLoader:
    """
    Loads an image file and returns a BGR uint8 numpy array.

    Usage
    -----
    image = ImageLoader.load("/path/to/image.jpg")
    info  = ImageLoader.info("/path/to/image.jpg")
    """

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        """
        Load an image from disk.

        Tries OpenCV first; falls back to Pillow if cv2.imread returns None
        (e.g., for unusual encodings or file-name characters).

        Parameters
        ----------
        path : str or Path
            Absolute or relative path to the image file.

        Returns
        -------
        np.ndarray
            BGR uint8 image of shape (H, W, 3).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        OSError
            If the file exists but cannot be decoded as an image.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # --- Primary: OpenCV ---
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

        # --- Fallback: Pillow ---
        logger.debug("cv2.imread failed for %s — trying PIL", path.name)
        try:
            pil_img = Image.open(path).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except (UnidentifiedImageError, Exception) as exc:
            raise OSError(f"Cannot decode image: {path}") from exc

    @staticmethod
    def load_grayscale(path: str | Path) -> np.ndarray:
        """
        Load an image as grayscale.

        Returns
        -------
        np.ndarray
            Single-channel uint8 image of shape (H, W).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image

        # Pillow fallback
        try:
            pil_img = Image.open(path).convert("L")
            return np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            raise OSError(f"Cannot decode image: {path}") from exc

    @staticmethod
    def info(path: str | Path) -> dict:
        """
        Return basic metadata about an image file without fully decoding it.

        Returns
        -------
        dict with keys: width, height, mode, format, size_bytes
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            with Image.open(path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "size_bytes": path.stat().st_size,
                }
        except Exception as exc:
            raise OSError(f"Cannot read image info: {path}") from exc

    @staticmethod
    def is_supported(path: str | Path) -> bool:
        """Return True if the file extension is in the supported list."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
