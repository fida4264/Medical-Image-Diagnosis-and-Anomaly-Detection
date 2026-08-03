"""
augmentation/policies/retina_policy.py
========================================
Augmentation policy for Retinal Fundus images.

Medical Safety Rules
--------------------
- Full 360° rotation is valid (no fixed orientation for retinal images)
- Horizontal flip is valid (left/right eye view)
- NO vertical flip (disc/fovea positioning would be anatomically wrong)
- Color jitter (simulate different fundus camera settings)
- Brightness/contrast (illumination variation)
- Small crop (simulate different FOV settings)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class RetinaAugmenter(BaseAugmenter):
    """Augmentation policy for Retinal Fundus modality."""

    @property
    def modality(self) -> str:
        return "retina"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Retina augmentation pipeline.

        Transforms:
        - Full rotation (retinal images have no fixed orientation)
        - Horizontal flip (left/right eye)
        - Brightness/contrast (fundus camera variation)
        - Hue/saturation/value jitter (color calibration variation)
        - Random crop (field-of-view variation)
        - Gaussian noise
        """
        mod = config.get("modalities", {}).get("retina", {})
        com = config.get("common", {})

        rot_limit = mod.get("rotation", {}).get("limit", 180)
        hflip_p = mod.get("horizontal_flip", {}).get("probability", 0.5)
        hflip_en = mod.get("horizontal_flip", {}).get("enabled", True)

        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.2)
        bc_con = bc.get("contrast_limit", 0.2)
        bc_p = bc.get("probability", 0.7)

        cj = mod.get("color_jitter", {})
        cj_enabled = cj.get("enabled", True)
        cj_hue = cj.get("hue_shift_limit", 10)
        cj_sat = cj.get("sat_shift_limit", 20)
        cj_val = cj.get("val_shift_limit", 20)
        cj_p = cj.get("probability", 0.5)

        crop_cfg = mod.get("crop", com.get("crop", {}))
        crop_enabled = crop_cfg.get("enabled", True)
        crop_min_max = crop_cfg.get("min_max_height", [180, 224])
        crop_p = crop_cfg.get("probability", 0.3)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 20])
        noise_p = noise.get("probability", 0.3)

        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.4)

        transforms = [
            A.Rotate(limit=rot_limit, p=0.7),
        ]

        if hflip_en:
            transforms.append(A.HorizontalFlip(p=hflip_p))

        transforms += [
            A.RandomBrightnessContrast(
                brightness_limit=bc_brt,
                contrast_limit=bc_con,
                p=bc_p,
            ),
            A.RandomGamma(gamma_limit=gamma_limit, p=gamma_p),
        ]

        if cj_enabled:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=cj_hue,
                    sat_shift_limit=cj_sat,
                    val_shift_limit=cj_val,
                    p=cj_p,
                )
            )

        if crop_enabled:
            transforms.append(
                A.RandomSizedCrop(
                    min_max_height=crop_min_max,
                    size=(224, 224),
                    p=crop_p,
                )
            )

        transforms.append(
            A.GaussNoise(
                std_range=(noise_var[0] / 255.0, noise_var[1] / 255.0),
                p=noise_p,
            )
        )

        return A.Compose(transforms)
