"""
preprocessing/pipelines/brain_mri_preprocessor.py
====================================================
Preprocessing pipeline for Brain MRI images.

Pipeline
--------
1. Convert to grayscale (MRI is always single-channel)
2. Normalize intensity to [0, 255]
3. Apply CLAHE (contrast enhancement)
4. Apply Gaussian denoising
5. Replicate to 3 channels (for CNN compatibility)
6. Resize with aspect-ratio-preserving padding

Optional (interface stubs for future integration):
- Skull stripping
- Bias field correction

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_clahe,
    apply_gaussian_denoise,
    convert_to_grayscale,
    ensure_3_channel,
    normalize_to_uint8,
    resize_with_padding,
)


class BrainMRIPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Brain MRI modality."""

    @property
    def modality(self) -> str:
        return "brain_mri"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Apply Brain MRI preprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 input image.

        Returns
        -------
        processed : np.ndarray
            BGR uint8 image of shape (H, W, 3).
        operations : list of str
            Operations applied in order.
        """
        ops: List[str] = []
        cfg = self.modality_config

        # 1. Convert to grayscale
        processed = convert_to_grayscale(image)
        ops.append("convert_to_grayscale")

        # 2. Intensity normalization (stretch to full range)
        processed = normalize_to_uint8(processed)
        ops.append("intensity_normalize")

        # 3. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 4. Gaussian denoising
        denoise_cfg = cfg.get("gaussian_denoise", {})
        if denoise_cfg.get("enabled", True):
            ksize = int(denoise_cfg.get("kernel_size", 3))
            sigma = float(denoise_cfg.get("sigma", 1.0))
            processed = apply_gaussian_denoise(processed, kernel_size=ksize, sigma=sigma)
            ops.append(f"gaussian_denoise(k={ksize},sigma={sigma})")

        # 5. Skull stripping interface (stub)
        if cfg.get("skull_strip_interface", False):
            ops.append("skull_strip[stub]")

        # 6. Bias field correction interface (stub)
        if cfg.get("bias_field_correction", False):
            ops.append("bias_field_correction[stub]")

        # 7. Ensure 3 channels
        processed = ensure_3_channel(processed)
        ops.append("ensure_3_channel")

        # 8. Resize with padding
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
