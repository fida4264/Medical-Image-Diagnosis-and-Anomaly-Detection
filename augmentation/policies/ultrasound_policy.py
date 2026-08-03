"""
augmentation/policies/ultrasound_policy.py
============================================
Augmentation policy for Ultrasound images.

Medical Safety Rules
--------------------
- NO vertical flip (probe orientation defines image top)
- NO horizontal flip (depth axis is vertical — flipping changes anatomy)
- Small rotation (≤15°) — probe angle variation
- Speckle noise (inherent to ultrasound physics)
- Brightness/contrast (gain setting variation)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class UltrasoundAugmenter(BaseAugmenter):
    """Augmentation policy for Ultrasound modality."""

    @property
    def modality(self) -> str:
        return "ultrasound"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Ultrasound augmentation pipeline.

        Transforms:
        - Small rotation (probe angulation)
        - Small shift/scale
        - Brightness/contrast (gain simulation)
        - Gaussian noise (speckle simulation)
        - Gamma variation
        - NO flips
        """
        mod = config.get("modalities", {}).get("ultrasound", {})
        com = config.get("common", {})

        rot_limit = mod.get("rotation", {}).get("limit", 15)

        ssr = mod.get("shift_scale_rotate", com.get("shift_scale_rotate", {}))
        ssr_shift = ssr.get("shift_limit", 0.04)
        ssr_scale = ssr.get("scale_limit", 0.04)
        ssr_rot = ssr.get("rotate_limit", 12)
        ssr_p = ssr.get("probability", 0.45)

        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.1)
        bc_con = bc.get("contrast_limit", 0.15)
        bc_p = bc.get("probability", 0.6)

        speckle = mod.get("speckle_noise", {})
        speckle_enabled = speckle.get("enabled", True)
        speckle_var = speckle.get("var_limit", [5, 25])
        speckle_p = speckle.get("probability", 0.5)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 20])
        noise_p = noise.get("probability", 0.4)

        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.35)

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
            A.RandomGamma(gamma_limit=gamma_limit, p=gamma_p),
        ]

        # Speckle noise — use GaussNoise as speckle approximation
        if speckle_enabled:
            transforms.append(
                A.GaussNoise(
                    std_range=(speckle_var[0] / 255.0, speckle_var[1] / 255.0),
                    p=speckle_p,
                )
            )

        transforms.append(
            A.GaussNoise(
                std_range=(noise_var[0] / 255.0, noise_var[1] / 255.0),
                p=noise_p,
            )
        )

        # Blur simulates ultrasound speckle texture
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.2))

        return A.Compose(transforms)
