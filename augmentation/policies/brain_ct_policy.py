"""
augmentation/policies/brain_ct_policy.py
==========================================
Augmentation policy for Brain CT images.

Medical Safety Rules
--------------------
- NO vertical flip
- NO horizontal flip (brain anatomy is asymmetric)
- Very small rotation (≤10°) — CT scanners are well-calibrated
- Brightness/contrast within CT window range
- Gaussian noise to simulate CT reconstruction variability
- Window-level variation (simulates different display settings)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class BrainCTAugmenter(BaseAugmenter):
    """Augmentation policy for Brain CT modality."""

    @property
    def modality(self) -> str:
        return "brain_ct"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Brain CT augmentation pipeline.

        Transforms:
        - Small rotation (≤10°)
        - Small shift/scale
        - Brightness/contrast (simulate window variations)
        - Gaussian noise (CT reconstruction noise)
        - Gamma (simulate brightness perception differences)
        - NO flips
        """
        mod = config.get("modalities", {}).get("brain_ct", {})
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
        ssr_shift = ssr.get("shift_limit", 0.03)
        ssr_scale = ssr.get("scale_limit", 0.03)
        ssr_rot = ssr.get("rotate_limit", 8)
        ssr_p = ssr.get("probability", 0.5)

        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.1)
        bc_con = bc.get("contrast_limit", 0.2)
        bc_p = bc.get("probability", 0.6)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 25])
        noise_p = noise.get("probability", 0.4)

        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.3)

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
            A.RandomGamma(
                gamma_limit=gamma_limit,
                p=gamma_p,
            ),
            # Equalization to simulate different display windows
            A.Equalize(p=0.2),
        ]

        return A.Compose(transforms)
