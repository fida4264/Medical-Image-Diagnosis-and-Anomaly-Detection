"""
preprocessing/pipelines/skin_preprocessor.py
==============================================
Preprocessing pipeline for Skin Lesion images.

Pipeline
--------
1. Color normalization (per-channel min-max)
2. CLAHE in LAB space
3. Normalize intensity to [0, 255]
4. Resize with padding

Optional (interface stubs):
- Hair artifact removal

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_clahe,
    apply_color_normalize,
    normalize_to_uint8,
    resize_with_padding,
)


class SkinPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Skin Lesion modality."""

    @property
    def modality(self) -> str:
        return "skin"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        processed = image.copy()

        # 1. Hair artifact removal interface (stub)
        if cfg.get("hair_removal_interface", False):
            ops.append("hair_removal[stub]")

        # 2. Color normalization
        color_cfg = cfg.get("color_normalize", {})
        c_enabled = color_cfg if isinstance(color_cfg, bool) else color_cfg.get("enabled", True)
        if c_enabled and len(image.shape) == 3:
            processed = apply_color_normalize(processed)
            ops.append("color_normalize")

        # 3. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 4. Intensity normalization
        if cfg.get("intensity_normalize", True):
            processed = normalize_to_uint8(processed)
            ops.append("intensity_normalize")

        # 5. Resize
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
