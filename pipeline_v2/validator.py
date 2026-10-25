"""
pipeline_v2/validator.py
========================
Strict input validation and sanitization for medical imaging inputs.
Enforces format, file size, dimension, corruption, and modality compatibility checks.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

from pipeline_v2.config import (
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_DIM,
    MAX_UPLOAD_SIZE_BYTES,
    MIN_IMAGE_DIM,
    SUPPORTED_MODALITIES,
)


class InputValidationError(Exception):
    """Raised when an input image or modality fails validation checks."""
    def __init__(self, message: str, error_code: str = "INVALID_INPUT"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class InputValidator:
    """
    Validates medical image inputs and modality requests before running inference.
    """

    @staticmethod
    def validate_modality(modality: Optional[str]) -> str:
        """
        Validates the requested modality.
        Rejects unsupported or missing modalities without guessing with unjustified certainty.
        """
        if not modality:
            raise InputValidationError(
                "Modality must be explicitly specified. Automatic appearance-only guessing is disabled "
                "to prevent cross-modality diagnostic misrouting.",
                error_code="MISSING_MODALITY",
            )
        mod_clean = str(modality).strip().lower()
        if mod_clean not in SUPPORTED_MODALITIES:
            raise InputValidationError(
                f"Modality '{modality}' is not supported. Supported modalities: {SUPPORTED_MODALITIES}",
                error_code="UNSUPPORTED_MODALITY",
            )
        return mod_clean

    @classmethod
    def validate_image_input(
        cls,
        image_input: Union[bytes, str, Path, Image.Image, io.BytesIO],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Validates and decodes image data into a clean PIL RGB Image.

        Args:
            image_input: Raw image bytes, file path, PIL Image, or BytesIO.

        Returns:
            Tuple of (sanitized PIL RGB Image, metadata dict).

        Raises:
            InputValidationError if any validation check fails.
        """
        meta: Dict[str, Any] = {
            "source_type": type(image_input).__name__,
        }

        # 1. Handle PIL Image directly
        if isinstance(image_input, Image.Image):
            pil_img = image_input
            meta["original_size"] = pil_img.size
            meta["original_mode"] = pil_img.mode
            meta["file_size_bytes"] = None

        # 2. Handle File Path
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise InputValidationError(f"Image file does not exist: {path}", error_code="FILE_NOT_FOUND")
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise InputValidationError(
                    f"File extension '{ext}' is not allowed. Supported formats: {sorted(ALLOWED_EXTENSIONS)}",
                    error_code="UNSUPPORTED_EXTENSION",
                )
            file_size = path.stat().st_size
            if file_size == 0:
                raise InputValidationError("Image file is empty (0 bytes).", error_code="EMPTY_FILE")
            if file_size > MAX_UPLOAD_SIZE_BYTES:
                raise InputValidationError(
                    f"File size ({file_size / (1024*1024):.1f}MB) exceeds limit of 20MB.",
                    error_code="FILE_TOO_LARGE",
                )
            meta["file_path"] = str(path)
            meta["file_size_bytes"] = file_size
            try:
                with open(path, "rb") as f:
                    raw_bytes = f.read()
                # Verify header and pixel decoding
                with Image.open(io.BytesIO(raw_bytes)) as test_img:
                    test_img.verify()
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img.load()
            except Exception as e:
                raise InputValidationError(f"Image corrupted or unreadable: {e}", error_code="CORRUPT_IMAGE")

        # 3. Handle Bytes / BytesIO
        elif isinstance(image_input, (bytes, io.BytesIO)):
            raw_bytes = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
            file_size = len(raw_bytes)
            if file_size == 0:
                raise InputValidationError("Uploaded image bytes are empty (0 bytes).", error_code="EMPTY_FILE")
            if file_size > MAX_UPLOAD_SIZE_BYTES:
                raise InputValidationError(
                    f"Image payload ({file_size / (1024*1024):.1f}MB) exceeds limit of 20MB.",
                    error_code="FILE_TOO_LARGE",
                )
            meta["file_size_bytes"] = file_size
            try:
                with Image.open(io.BytesIO(raw_bytes)) as test_img:
                    test_img.verify()
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img.load()
            except Exception as e:
                raise InputValidationError(f"Image corrupted or unreadable: {e}", error_code="CORRUPT_IMAGE")

        else:
            raise InputValidationError(
                f"Unsupported image input type: {type(image_input)}",
                error_code="UNSUPPORTED_INPUT_TYPE",
            )

        # 4. Dimension & Resolution Checks
        width, height = pil_img.size
        meta["width"] = width
        meta["height"] = height
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            raise InputValidationError(
                f"Image dimensions ({width}x{height}) are below minimum supported resolution ({MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}).",
                error_code="IMAGE_TOO_SMALL",
            )
        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            raise InputValidationError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({MAX_IMAGE_DIM}px).",
                error_code="IMAGE_TOO_LARGE",
            )

        # 5. Channel Sanitization (Convert grayscale/RGBA to standard 3-channel RGB)
        if pil_img.mode != "RGB":
            meta["original_mode"] = pil_img.mode
            sanitized_img = pil_img.convert("RGB")
        else:
            sanitized_img = pil_img

        return sanitized_img, meta
