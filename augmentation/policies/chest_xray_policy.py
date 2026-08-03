"""
augmentation/policies/chest_xray_policy.py
============================================
Augmentation policy for Chest X-ray images.

Medical Safety Rules
--------------------
- NO vertical flip (heart/lung anatomy positioning is fixed)
- NO horizontal flip (left/right heart anatomy must be preserved)
- Small rotation (≤10°) — patient positioning variability
- Brightness/contrast variation (different X-ray exposure settings)
- Histogram variation (different display LUT settings)
- Small translation (patient centering variability)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class ChestXRayAugmenter(BaseAugmenter):
    """Augmentation policy for Chest X-ray modality."""

    @property
    def modality(self) -> str:
        return "chest_xray"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Chest X-ray augmentation pipeline.

        Transforms:
        - Small rotation (simulate patient tilt)
        - Small shift (simulate patient centering variation)
        - Brightness/contrast (simulate exposure settings)
        - Histogram equalization variation
        - Gaussian noise
        - Gamma variation
        - NO flips (cardiac silhouette must stay left)
        """
        mod = config.get("modalities", {}).get("chest_xray", {})
        com = config.get("common", {})

        def _get(section, key, default):
            m = mod.get(section, {})
            c = com.get(section, {})
            if isinstance(m, dict) and key in m:
                return m[key]
            if isinstance(c, dict) and key in c:
                return c[key]
            return default

        rot_limit = _get("rotation", "limit", 10)

        ssr = mod.get("shift_scale_rotate", com.get("shift_scale_rotate", {}))
        ssr_shift = ssr.get("shift_limit", 0.05)
        ssr_scale = ssr.get("scale_limit", 0.05)
        ssr_rot = ssr.get("rotate_limit", 8)
        ssr_p = ssr.get("probability", 0.5)

        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.15)
        bc_con = bc.get("contrast_limit", 0.2)
        bc_p = bc.get("probability", 0.7)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 25])
        noise_p = noise.get("probability", 0.35)

        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.35)

        hist_cfg = mod.get("histogram_variation", {})
        hist_p = hist_cfg.get("probability", 0.3)

        transforms = [
            A.Rotate(limit=rot_limit, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=ssr_shift,
                scale_limit=ssr_scale,
                rotate_limit=ssr_rot,
                p=ssr_p,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=bc_brt,
                contrast_limit=bc_con,
                p=bc_p,
            ),
            A.GaussNoise(
                std_range=(noise_var[0] / 255.0, noise_var[1] / 255.0),
                p=noise_p,
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=gamma_p),
            A.Equalize(p=hist_p),
            A.CLAHE(clip_limit=(1.0, 3.0), p=0.3),
        ]

        return A.Compose(transforms)
