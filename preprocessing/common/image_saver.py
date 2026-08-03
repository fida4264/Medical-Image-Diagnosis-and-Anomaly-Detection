"""
preprocessing/common/image_saver.py
======================================
Saves processed images to disk in a consistent format.

Always saves as PNG (lossless) unless the config explicitly requests JPEG.
Converts float32 arrays to uint8 before saving.

Classes
-------
ImageSaver
    Static methods for saving images and creating output directories.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageSaver:
    """
    Saves BGR uint8 numpy arrays to image files on disk.

    Usage
    -----
    ImageSaver.save(processed_image, Path("processed/brain_mri/glioma/MRI_000001.png"))
    """

    @staticmethod
    def save(
        image: np.ndarray,
        output_path: str | Path,
        quality: int = 95,
    ) -> Path:
        """
        Save an image to disk.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 numpy array.  Float arrays in [0,1] or [0,255] are
            automatically converted.
        output_path : str or Path
            Destination file path.  Parent directories are created if missing.
        quality : int
            JPEG quality (1-100, default 95).  Ignored for PNG.

        Returns
        -------
        Path
            Path of the saved file.

        Raises
        ------
        IOError
            If cv2.imwrite fails (e.g., disk full or bad path).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure uint8
        image = ImageSaver._to_uint8(image)

        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]   # moderate compression
        else:
            params = []

        success = cv2.imwrite(str(output_path), image, params)
        if not success:
            raise IOError(f"cv2.imwrite failed for path: {output_path}")

        return output_path

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Convert float or non-uint8 image to uint8."""
        if image.dtype == np.uint8:
            return image
        if image.dtype in (np.float32, np.float64):
            max_val = image.max()
            if max_val <= 1.0:
                image = image * 255.0
            return image.clip(0, 255).astype(np.uint8)
        return image.astype(np.uint8)

    @staticmethod
    def ensure_output_dirs(base_dir: Path, modalities: list) -> None:
        """Create the modality output subdirectories."""
        for mod in modalities:
            (base_dir / mod).mkdir(parents=True, exist_ok=True)
