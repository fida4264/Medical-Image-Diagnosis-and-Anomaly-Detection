"""
image_utils.py
==============
Shared, low-level image utility functions used across every module that handles
medical images.

Design principle: every function here is pure, stateless, and side-effect-free.
They accept a path (or bytes) and return a result. No global state. No logging
side-effects beyond what the caller controls.

Functions
---------
is_supported_image          Check whether a file extension is in the allow-list.
is_hidden_file              Check whether a file/folder starts with a dot.
load_image_safe             Load an image with PIL; returns None on any error.
get_image_dimensions        Return (width, height) of an image file safely.
is_corrupted                Return True if the image cannot be decoded.
compute_md5_hash            Return the hex MD5 digest of a file's raw bytes.
get_file_size_bytes         Return the file size in bytes.

Author: Medical AI System — Module 1
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

from dataset_manager.config import (
    MIN_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File / Path Helpers
# ---------------------------------------------------------------------------

def is_supported_image(path: Path) -> bool:
    """
    Return True if the file's extension is in the supported image formats list.

    Parameters
    ----------
    path : Path
        File path to evaluate.

    Returns
    -------
    bool
        True when the extension is supported; False otherwise.

    Examples
    --------
    >>> is_supported_image(Path("image.jpg"))
    True
    >>> is_supported_image(Path("report.pdf"))
    False
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_hidden_file(path: Path) -> bool:
    """
    Return True if any component of the path starts with a dot.

    This catches both hidden files (`.DS_Store`) and files inside hidden
    directories (`__MACOSX/image.png`).

    Parameters
    ----------
    path : Path
        File or directory path to evaluate.

    Returns
    -------
    bool
        True when the path contains a hidden component; False otherwise.
    """
    return any(part.startswith(".") for part in path.parts)


def get_file_size_bytes(path: Path) -> int:
    """
    Return the size of a file in bytes.

    Parameters
    ----------
    path : Path
        Path to an existing file.

    Returns
    -------
    int
        File size in bytes; 0 if the file cannot be accessed.
    """
    try:
        return path.stat().st_size
    except OSError as exc:
        logger.debug("Cannot read file size for %s: %s", path, exc)
        return 0


# ---------------------------------------------------------------------------
# Image Loading and Inspection
# ---------------------------------------------------------------------------

def load_image_safe(path: Path) -> Optional[Image.Image]:
    """
    Attempt to open an image file with PIL.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    PIL.Image.Image or None
        The loaded image object, or None if loading failed for any reason.
    """
    try:
        img = Image.open(path)
        img.load()          # Force full decode to catch truncated images
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as exc:
        logger.debug("Failed to load image %s: %s", path, exc)
        return None


def get_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """
    Return the (width, height) of an image in pixels.

    This uses PIL's lazy header-only read, which avoids decoding the full
    image data and is much faster for large datasets.

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    tuple of (int, int) or None
        (width, height) on success; None if the file cannot be read.
    """
    try:
        with Image.open(path) as img:
            return img.size     # PIL returns (width, height)
    except (UnidentifiedImageError, OSError, SyntaxError, Exception) as exc:
        logger.debug("Cannot read dimensions for %s: %s", path, exc)
        return None


def is_corrupted(path: Path) -> bool:
    """
    Return True if the image file is unreadable or corrupted.

    The check has three levels of increasing strictness:
    1. File must have a non-trivial size (> MIN_FILE_SIZE_BYTES).
    2. PIL must be able to identify the file as an image.
    3. PIL must be able to fully decode the pixel data (catches truncated files).

    Parameters
    ----------
    path : Path
        Path to the image file.

    Returns
    -------
    bool
        True when the image is considered corrupted; False if it is healthy.
    """
    # Level 1 — size check (very fast, no I/O beyond stat)
    if get_file_size_bytes(path) < MIN_FILE_SIZE_BYTES:
        logger.debug("File too small, likely empty/corrupted: %s", path)
        return True

    # Level 2 & 3 — decode check
    image = load_image_safe(path)
    return image is None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_md5_hash(path: Path) -> str:
    """
    Compute the MD5 hex digest of a file's raw bytes.

    Used for duplicate-image detection across the dataset. MD5 is chosen for
    speed; this is not a security-sensitive context.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        Hex-encoded MD5 digest string (32 characters) on success;
        empty string on I/O error.
    """
    hasher = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            # Read in chunks to handle very large images efficiently
            for chunk in iter(lambda: fh.read(65_536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as exc:
        logger.debug("Cannot hash file %s: %s", path, exc)
        return ""
