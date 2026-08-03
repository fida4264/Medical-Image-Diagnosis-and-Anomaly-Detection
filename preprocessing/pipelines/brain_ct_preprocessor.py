"""
preprocessing/pipelines/brain_ct_preprocessor.py
===================================================
Preprocessing pipeline for Brain CT images.

Pipeline
--------
1. Convert to grayscale
2. Apply CT Hounsfield windowing (brain window: WL=40, WW=80)
3. Normalize intensity to [0, 255]
4. Apply CLAHE
5. Gaussian noise reduction
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
    apply_ct_windowing,
    convert_to_grayscale,
    ensure_3_channel,
    normalize_to_uint8,
    resize_with_padding,
)


class BrainCTPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Brain CT modality."""

    @property
    def modality(self) -> str:
        return "brain_ct"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        # 1. Convert to grayscale
        processed = convert_to_grayscale(image)
        ops.append("convert_to_grayscale")

        # 2. CT windowing
        window_cfg = cfg.get("windowing", {})
        if window_cfg.get("enabled", True):
            wl = int(window_cfg.get("window_level", 40))
            ww = int(window_cfg.get("window_width", 80))
            processed = apply_ct_windowing(processed, window_level=wl, window_width=ww)
            ops.append(f"ct_windowing(wl={wl},ww={ww})")
        else:
            # Fallback: simple normalization
            processed = normalize_to_uint8(processed)
            ops.append("intensity_normalize")

        # 3. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 4. Noise reduction
        noise_cfg = cfg.get("noise_reduction", {})
        if noise_cfg.get("enabled", True):
            ksize = int(noise_cfg.get("kernel_size", 3))
            processed = apply_gaussian_denoise(processed, kernel_size=ksize)
            ops.append(f"gaussian_denoise(k={ksize})")

        # 5. Ensure 3 channels
        processed = ensure_3_channel(processed)
        ops.append("ensure_3_channel")

        # 6. Resize
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
