"""
preprocessing/pipelines/retina_preprocessor.py
=================================================
Preprocessing pipeline for Retinal Fundus images.

Pipeline
--------
1. Circular crop — remove black border around fundus
2. Enhance green channel (carries most diagnostic detail)
3. Apply CLAHE in LAB space
4. Brightness normalization
5. Normalize intensity to [0, 255]
6. Resize with padding

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_circular_crop,
    apply_clahe,
    apply_color_normalize,
    enhance_green_channel,
    normalize_to_uint8,
    resize_with_padding,
)


class RetinaPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Retinal Fundus modality."""

    @property
    def modality(self) -> str:
        return "retina"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        # 1. Circular crop (remove black borders)
        crop_cfg = cfg.get("circular_crop", {})
        enabled = crop_cfg if isinstance(crop_cfg, bool) else crop_cfg.get("enabled", True)
        if enabled:
            threshold = (
                int(crop_cfg.get("border_threshold", 10))
                if isinstance(crop_cfg, dict)
                else 10
            )
            processed = apply_circular_crop(image, border_threshold=threshold)
            ops.append(f"circular_crop(threshold={threshold})")
        else:
            processed = image.copy()

        # 2. Green channel enhancement
        green_cfg = cfg.get("green_channel_enhance", {})
        g_enabled = green_cfg if isinstance(green_cfg, bool) else green_cfg.get("enabled", True)
        if g_enabled:
            factor = (
                float(green_cfg.get("enhance_factor", 1.2))
                if isinstance(green_cfg, dict)
                else 1.2
            )
            processed = enhance_green_channel(processed, factor=factor)
            ops.append(f"green_enhance(factor={factor})")

        # 3. CLAHE
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            processed = apply_clahe(processed, clip_limit=clip, tile_grid_size=grid)
            ops.append(f"clahe(clip={clip},grid={grid})")

        # 4. Brightness normalization (channel-wise)
        if cfg.get("brightness_normalize", True):
            processed = apply_color_normalize(processed)
            ops.append("brightness_normalize")

        # 5. Intensity normalization
        if cfg.get("intensity_normalize", True):
            processed = normalize_to_uint8(processed)
            ops.append("intensity_normalize")

        # 6. Resize
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
