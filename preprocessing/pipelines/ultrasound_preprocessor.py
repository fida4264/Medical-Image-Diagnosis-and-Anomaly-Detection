"""
preprocessing/pipelines/ultrasound_preprocessor.py
=====================================================
Preprocessing pipeline for Ultrasound images.

Pipeline
--------
1. Convert to grayscale (ultrasound is single-channel)
2. Normalize intensity to [0, 255]
3. Median filter for speckle noise reduction
4. Apply CLAHE
5. Replicate to 3 channels
6. Resize with padding

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_clahe,
    apply_median_filter,
    apply_bilateral_filter,
    convert_to_grayscale,
    ensure_3_channel,
    normalize_to_uint8,
    resize_with_padding,
)


class UltrasoundPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Ultrasound modality."""

    @property
    def modality(self) -> str:
        return "ultrasound"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        # 1. Convert to grayscale
        processed = convert_to_grayscale(image)
        ops.append("convert_to_grayscale")

        # 2. Normalize intensity
        processed = normalize_to_uint8(processed)
        ops.append("intensity_normalize")

        # 3. Speckle noise reduction
        speckle_cfg = cfg.get("speckle_reduction", {})
        if speckle_cfg.get("enabled", True):
            method = speckle_cfg.get("method", "median")
            ksize = int(speckle_cfg.get("kernel_size", 3))
            if method == "median":
                processed = apply_median_filter(processed, kernel_size=ksize)
                ops.append(f"median_filter(k={ksize})")
            elif method == "bilateral":
                processed = apply_bilateral_filter(processed)
                ops.append("bilateral_filter")
            else:
                processed = apply_median_filter(processed, kernel_size=ksize)
                ops.append(f"median_filter(k={ksize})")

        # 4. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 2.5))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 5. Ensure 3 channels
        processed = ensure_3_channel(processed)
        ops.append("ensure_3_channel")

        # 6. Resize
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
