"""
preprocessing/common/utils.py
===============================
Shared image processing utilities used by all modality pipelines.

All functions operate on BGR uint8 numpy arrays (OpenCV convention).
They are pure functions — no side effects, no state.

Functions
---------
apply_clahe               Apply CLAHE contrast enhancement.
apply_gaussian_denoise    Gaussian blur denoising.
apply_median_filter       Median filter (speckle / impulse noise).
apply_bilateral_filter    Edge-preserving bilateral filter.
normalize_to_uint8        Stretch intensity to full [0, 255] range.
resize_with_padding       Resize preserving aspect ratio; pad to square.
convert_to_grayscale      BGR → single-channel grayscale.
ensure_3_channel          Replicate grayscale channel → 3ch BGR.
apply_histogram_eq        Standard histogram equalization (grayscale).
apply_ct_windowing        CT Hounsfield Unit windowing.
apply_circular_crop       Remove circular black border (retina fundus).
apply_color_normalize     Per-channel min-max normalization (RGB images).
enhance_green_channel     Amplify green channel (retinal imaging).

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrast Enhancement
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    For grayscale images the transform is applied directly.
    For colour images CLAHE is applied to the L channel in LAB space to
    avoid colour shift.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 or grayscale uint8 input image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple of int
        Size of grid for histogram equalization.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image, same shape and dtype as input.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        return clahe.apply(gray)
    # Colour: operate in LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def apply_gaussian_denoise(
    image: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter — effective for speckle and salt-and-pepper noise."""
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral filter."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Intensity Normalization
# ---------------------------------------------------------------------------

def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Stretch pixel intensities to fill [0, 255].

    Works on grayscale and colour images.  If the image has zero range
    (all pixels the same value) it is returned unchanged.
    """
    img_f = image.astype(np.float32)
    min_v, max_v = img_f.min(), img_f.max()
    if max_v > min_v:
        img_f = (img_f - min_v) / (max_v - min_v) * 255.0
    return img_f.clip(0, 255).astype(np.uint8)


def apply_ct_windowing(
    image: np.ndarray,
    window_level: int = 40,
    window_width: int = 80,
) -> np.ndarray:
    """
    Apply CT Hounsfield Unit windowing.

    Maps the [WL - WW/2, WL + WW/2] intensity range to [0, 255].
    Pixels outside the window are clipped.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image (values already in 0-255 range after DICOM decode).
    window_level : int
        Centre of the Hounsfield window.
    window_width : int
        Width of the Hounsfield window.
    """
    lower = window_level - window_width // 2
    upper = window_level + window_width // 2
    img_f = image.astype(np.float32)
    img_f = np.clip(img_f, lower, upper)
    img_f = (img_f - lower) / max(upper - lower, 1) * 255.0
    return img_f.astype(np.uint8)


def apply_histogram_eq(image: np.ndarray) -> np.ndarray:
    """Apply standard histogram equalization to a grayscale image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return cv2.equalizeHist(image)


# ---------------------------------------------------------------------------
# Channel Operations
# ---------------------------------------------------------------------------

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_3_channel(image: np.ndarray) -> np.ndarray:
    """Replicate a grayscale image to a 3-channel BGR array."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0,
) -> np.ndarray:
    """
    Resize image preserving aspect ratio, padding shorter side to square.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C).
    target_size : tuple of int
        (target_height, target_width).
    pad_value : int
        Padding pixel value (default: 0 = black).

    Returns
    -------
    np.ndarray
        Resized and padded image of exactly (target_h, target_w[, C]).
    """
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if len(image.shape) == 3:
        canvas = np.full((th, tw, image.shape[2]), pad_value, dtype=np.uint8)
    else:
        canvas = np.full((th, tw), pad_value, dtype=np.uint8)

    y0 = (th - new_h) // 2
    x0 = (tw - new_w) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Colour & Retinal Utilities
# ---------------------------------------------------------------------------

def apply_color_normalize(image: np.ndarray) -> np.ndarray:
    """
    Per-channel min-max normalization for BGR colour images.

    Normalizes each channel independently to [0, 255].
    """
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        ch = image[:, :, c].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            out[:, :, c] = (ch - mn) / (mx - mn) * 255.0
        else:
            out[:, :, c] = ch
    return out.clip(0, 255).astype(np.uint8)


def enhance_green_channel(
    image: np.ndarray, factor: float = 1.2
) -> np.ndarray:
    """
    Enhance the green channel of a BGR image.

    The green channel carries the most diagnostic information in retinal
    fundus photography.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    factor : float
        Multiplicative scale applied to the green channel (default: 1.2).
    """
    out = image.astype(np.float32)
    out[:, :, 1] = np.clip(out[:, :, 1] * factor, 0, 255)
    return out.astype(np.uint8)


def apply_circular_crop(
    image: np.ndarray, border_threshold: int = 10
) -> np.ndarray:
    """
    Remove the circular black border from retinal fundus images.

    Finds the bounding box of the non-black region and crops to it.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    border_threshold : int
        Pixels with all channels below this value are treated as border.

    Returns
    -------
    np.ndarray
        Cropped image.  Returns original if no significant crop is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, border_threshold, 255, cv2.THRESH_BINARY)

    # Find bounding rect of the non-black region
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    # Add a small margin
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    cropped = image[y: y + h, x: x + w]
    if cropped.size == 0:
        return image
    return cropped


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Compute Laplacian variance — used as a blur/sharpness metric."""
    gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())
