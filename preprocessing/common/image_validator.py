"""
preprocessing/common/image_validator.py
=========================================
Pre- and post-processing image quality validation.

Detects seven classes of image issues:
  1. Corrupt / unloadable
  2. Wrong channel count
  3. Too small (below minimum dimension)
  4. Empty / all-black
  5. Too dark
  6. Too bright / overexposed
  7. Blurry / out of focus

Each check returns a QualityFlag string.  The validate() method
returns a list of all flags present in the image (empty = OK).

Classes
-------
ImageValidator
    Static methods for individual checks and batch validation.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np

from preprocessing.common.utils import compute_laplacian_variance, convert_to_grayscale

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flag constants
# ---------------------------------------------------------------------------

FLAG_BLURRY = "blurry"
FLAG_TOO_DARK = "too_dark"
FLAG_TOO_BRIGHT = "too_bright"
FLAG_TOO_SMALL = "too_small"
FLAG_EMPTY = "empty"
FLAG_BAD_CHANNELS = "bad_channels"
FLAG_CORRUPT = "corrupt"

ALL_FLAGS = {
    FLAG_BLURRY, FLAG_TOO_DARK, FLAG_TOO_BRIGHT,
    FLAG_TOO_SMALL, FLAG_EMPTY, FLAG_BAD_CHANNELS, FLAG_CORRUPT,
}

# Default thresholds (overridden by config when validate_with_config is used)
DEFAULT_BLUR_THRESHOLD = 80.0
DEFAULT_DARK_THRESHOLD = 20.0
DEFAULT_BRIGHT_THRESHOLD = 245.0
DEFAULT_MIN_DIMENSION = 32


class ImageValidator:
    """
    Image quality validator.

    All methods accept a BGR uint8 numpy array.

    Usage
    -----
    flags = ImageValidator.validate(image)
    if flags:
        print("Quality issues:", flags)
    """

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_blurry(
        image: np.ndarray,
        threshold: float = DEFAULT_BLUR_THRESHOLD,
    ) -> bool:
        """
        Return True if the image is blurry.

        Uses the variance of the Laplacian as a focus measure.
        A very low variance indicates little high-frequency detail (blur).
        """
        variance = compute_laplacian_variance(image)
        return variance < threshold

    @staticmethod
    def is_too_dark(
        image: np.ndarray,
        threshold: float = DEFAULT_DARK_THRESHOLD,
    ) -> bool:
        """Return True if the mean pixel intensity is below threshold."""
        gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
        return float(gray.mean()) < threshold

    @staticmethod
    def is_too_bright(
        image: np.ndarray,
        threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    ) -> bool:
        """Return True if the mean pixel intensity is above threshold."""
        gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
        return float(gray.mean()) > threshold

    @staticmethod
    def is_too_small(
        image: np.ndarray,
        min_dimension: int = DEFAULT_MIN_DIMENSION,
    ) -> bool:
        """Return True if either spatial dimension is below the minimum."""
        h, w = image.shape[:2]
        return h < min_dimension or w < min_dimension

    @staticmethod
    def is_empty(image: np.ndarray) -> bool:
        """Return True if the image is all-black (zero variance)."""
        return float(image.var()) < 1e-6

    @staticmethod
    def has_bad_channels(image: np.ndarray) -> bool:
        """Return True if the channel count is not 1 or 3."""
        if len(image.shape) == 2:
            return False  # valid grayscale
        c = image.shape[2]
        return c not in (1, 3)

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(
        image: np.ndarray,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        dark_threshold: float = DEFAULT_DARK_THRESHOLD,
        bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
        min_dimension: int = DEFAULT_MIN_DIMENSION,
    ) -> List[str]:
        """
        Run all quality checks and return a list of flags.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image array.

        Returns
        -------
        list of str
            Quality flags present.  Empty list means the image passes all
            checks.
        """
        flags: List[str] = []

        if ImageValidator.is_empty(image):
            flags.append(FLAG_EMPTY)
            return flags  # no point running other checks

        if ImageValidator.has_bad_channels(image):
            flags.append(FLAG_BAD_CHANNELS)

        if ImageValidator.is_too_small(image, min_dimension):
            flags.append(FLAG_TOO_SMALL)

        if ImageValidator.is_too_dark(image, dark_threshold):
            flags.append(FLAG_TOO_DARK)

        if ImageValidator.is_too_bright(image, bright_threshold):
            flags.append(FLAG_TOO_BRIGHT)

        if ImageValidator.is_blurry(image, blur_threshold):
            flags.append(FLAG_BLURRY)

        return flags

    @staticmethod
    def validate_with_config(
        image: np.ndarray,
        quality_config: Dict,
    ) -> List[str]:
        """
        Run all checks using thresholds from the YAML quality config section.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image array.
        quality_config : dict
            The ``quality:`` section of preprocessing.yaml.
        """
        return ImageValidator.validate(
            image,
            blur_threshold=float(quality_config.get("blur_threshold", DEFAULT_BLUR_THRESHOLD)),
            dark_threshold=float(quality_config.get("dark_threshold", DEFAULT_DARK_THRESHOLD)),
            bright_threshold=float(quality_config.get("bright_threshold", DEFAULT_BRIGHT_THRESHOLD)),
            min_dimension=int(quality_config.get("min_dimension", DEFAULT_MIN_DIMENSION)),
        )

    @staticmethod
    def summary(flags: List[str]) -> str:
        """Return a human-readable quality summary string."""
        if not flags:
            return "OK"
        return f"FLAGGED({','.join(sorted(flags))})"
