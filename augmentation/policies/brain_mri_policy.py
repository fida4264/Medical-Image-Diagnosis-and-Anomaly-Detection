"""
augmentation/policies/brain_mri_policy.py
===========================================
Augmentation policy for Brain MRI images.

Medical Safety Rules
--------------------
- NO vertical flip (brain orientation must be preserved)
- NO horizontal flip (left/right brain anatomy must be preserved)
- Small rotation only (≤15°)
- Small translation (≤5%)
- Brightness/contrast within safe medical range
- Gaussian noise to simulate MRI acquisition variability
- CLAHE variation to simulate scanner differences

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class BrainMRIAugmenter(BaseAugmenter):
    """Augmentation policy for Brain MRI modality."""

    @property
    def modality(self) -> str:
        return "brain_mri"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Brain MRI augmentation pipeline.

        Medically safe transforms only:
        - Small rotation and translation
        - Brightness / contrast variation
        - Gaussian noise (MRI noise simulation)
        - CLAHE variation (scanner contrast simulation)
        - NO flips
        """
        mod = config.get("modalities", {}).get("brain_mri", {})
        com = config.get("common", {})

        def _get(section, key, default):
            m = mod.get(section, {})
            c = com.get(section, {})
            if isinstance(m, dict) and key in m:
                return m[key]
            if isinstance(c, dict) and key in c:
                return c[key]
            return default

        # Rotation limit
        rot_limit = _get("rotation", "limit", 15)

        # Shift / scale / rotate
        ssr = mod.get("shift_scale_rotate", com.get("shift_scale_rotate", {}))
        ssr_enabled = ssr.get("enabled", True)
        ssr_shift = ssr.get("shift_limit", 0.03)
        ssr_scale = ssr.get("scale_limit", 0.03)
        ssr_rot = ssr.get("rotate_limit", 10)
        ssr_p = ssr.get("probability", 0.5)

        # Brightness / contrast
        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.1)
        bc_con = bc.get("contrast_limit", 0.15)
        bc_p = bc.get("probability", 0.6)

        # Noise
        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 20])
        noise_p = noise.get("probability", 0.4)

        # Gamma
        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.4)

        # CLAHE variation
        clahe_cfg = mod.get("clahe_variation", {})
        clahe_enabled = clahe_cfg.get("enabled", True)
        clahe_clip = clahe_cfg.get("clip_limit_range", [1.5, 3.0])
        clahe_p = clahe_cfg.get("probability", 0.4)

        transforms = [
            A.Rotate(limit=rot_limit, p=0.6),
        ]

        if ssr_enabled:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=ssr_shift,
                    scale_limit=ssr_scale,
                    rotate_limit=ssr_rot,
                    p=ssr_p,
                )
            )

        transforms += [
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
        ]

        if clahe_enabled:
            transforms.append(
                A.CLAHE(
                    clip_limit=clahe_clip,
                    p=clahe_p,
                )
            )

        return A.Compose(transforms)
