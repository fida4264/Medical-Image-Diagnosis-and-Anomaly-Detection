"""
preprocessing/pipelines/histopathology_preprocessor.py
=========================================================
Preprocessing pipeline for Histopathology (H&E stained) images.

Pipeline
--------
1. Color normalization (channel-wise)
2. Optional stain normalization (interface stub — requires reference image)
3. Normalize intensity to [0, 255]
4. Resize with padding (768x768 → 224x224)

Note: CLAHE is intentionally disabled for H&E slides as it distorts
      the diagnostic stain color information.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing.base.base_preprocessor import BasePreprocessor
from preprocessing.common.utils import (
    apply_color_normalize,
    normalize_to_uint8,
    resize_with_padding,
)


class HistopathologyPreprocessor(BasePreprocessor):
    """Preprocessing pipeline for Histopathology modality."""

    @property
    def modality(self) -> str:
        return "histopathology"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        ops: List[str] = []
        cfg = self.modality_config

        processed = image.copy()

        # 1. Color normalization
        color_cfg = cfg.get("color_normalize", {})
        c_enabled = color_cfg if isinstance(color_cfg, bool) else color_cfg.get("enabled", True)
        if c_enabled and len(image.shape) == 3:
            processed = apply_color_normalize(processed)
            ops.append("color_normalize")

        # 2. Stain normalization interface (stub)
        stain_cfg = cfg.get("stain_normalize", {})
        s_enabled = stain_cfg if isinstance(stain_cfg, bool) else stain_cfg.get("enabled", False)
        if s_enabled:
            method = stain_cfg.get("method", "macenko") if isinstance(stain_cfg, dict) else "macenko"
            # Stub: actual implementation requires a reference image and
            # stain separation libraries (e.g., staintools, torchstain)
            ops.append(f"stain_normalize[{method}_stub]")

        # 3. Intensity normalization
        if cfg.get("intensity_normalize", True):
            processed = normalize_to_uint8(processed)
            ops.append("intensity_normalize")

        # 4. Resize (768×768 → 224×224)
        target = self.target_size
        processed = resize_with_padding(processed, target)
        ops.append(f"resize({target[1]}x{target[0]})")

        return processed, ops
