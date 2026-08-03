"""
preprocessing/pipelines/chest_xray_preprocessor.py
=====================================================
Preprocessing pipeline for Chest X-Ray images.

Pipeline
--------
1. Convert to grayscale (X-rays are single-channel)
2. Normalize intensity to [0, 255]
3. Apply CLAHE (strong clip=3.0 — X-rays need aggressive enhancement)
4. Optional histogram equalization
5. Gaussian noise removal
6. Replicate to 3 channels
7. Resize with padding

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_clahe,
    apply_gaussian_denoise,
    apply_histogram_eq,
    convert_to_grayscale,
    ensure_3_channel,
    normalize_to_uint8,
    resize_with_padding,
)


class ChestXRayPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Chest X-Ray modality."""

    @property
    def modality(self) -> str:
        return "chest_xray"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        # 1. Convert to grayscale
        processed = convert_to_grayscale(image)
        ops.append("convert_to_grayscale")

        # 2. Normalize intensity
        processed = normalize_to_uint8(processed)
        ops.append("intensity_normalize")

        # 3. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 3.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 4. Optional histogram equalization (additional contrast)
        if cfg.get("histogram_equalization", False):
            processed = apply_histogram_eq(processed)
            ops.append("histogram_eq")

        # 5. Noise reduction
        noise_cfg = cfg.get("noise_reduction", {})
        if noise_cfg.get("enabled", True):
            ksize = int(noise_cfg.get("kernel_size", 3))
            processed = apply_gaussian_denoise(processed, kernel_size=ksize)
            ops.append(f"gaussian_denoise(k={ksize})")

        # 6. Ensure 3 channels
        processed = ensure_3_channel(processed)
        ops.append("ensure_3_channel")

        # 7. Resize
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
