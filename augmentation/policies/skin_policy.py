"""
augmentation/policies/skin_policy.py
======================================
Augmentation policy for Skin Lesion images.

Medical Safety Rules
--------------------
- Full 360° rotation valid (no fixed orientation for dermatoscopy)
- Horizontal AND vertical flip valid (lesions have no orientation)
- Color jitter (different dermatoscope / lighting settings)
- Brightness/contrast variation
- Zoom / random crop (magnification variation)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class SkinAugmenter(BaseAugmenter):
    """Augmentation policy for Skin Lesion modality."""

    @property
    def modality(self) -> str:
        return "skin"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Skin Lesion augmentation pipeline.

        Transforms:
        - Full rotation
        - Horizontal and vertical flip
        - Brightness / contrast
        - Color jitter (dermatoscope variation)
        - Zoom / random crop
        - Gaussian noise
        """
        mod = config.get("modalities", {}).get("skin", {})
        com = config.get("common", {})

        rot_limit = mod.get("rotation", {}).get("limit", 180)
        hflip_en = mod.get("horizontal_flip", {}).get("enabled", True)
        hflip_p = mod.get("horizontal_flip", {}).get("probability", 0.5)
        vflip_en = mod.get("vertical_flip", {}).get("enabled", True)
        vflip_p = mod.get("vertical_flip", {}).get("probability", 0.5)

        bc = mod.get("brightness_contrast", com.get("brightness_contrast", {}))
        bc_brt = bc.get("brightness_limit", 0.2)
        bc_con = bc.get("contrast_limit", 0.2)
        bc_p = bc.get("probability", 0.7)

        cj = mod.get("color_jitter", {})
        cj_enabled = cj.get("enabled", True)
        cj_hue = cj.get("hue_shift_limit", 15)
        cj_sat = cj.get("sat_shift_limit", 25)
        cj_val = cj.get("val_shift_limit", 25)
        cj_p = cj.get("probability", 0.5)

        zoom_cfg = mod.get("zoom", {})
        zoom_enabled = zoom_cfg.get("enabled", True)
        zoom_scale = zoom_cfg.get("scale_limit", 0.15)
        zoom_p = zoom_cfg.get("probability", 0.5)

        crop_cfg = mod.get("crop", com.get("crop", {}))
        crop_enabled = crop_cfg.get("enabled", True)
        crop_min_max = crop_cfg.get("min_max_height", [180, 224])
        crop_p = crop_cfg.get("probability", 0.4)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 30])
        noise_p = noise.get("probability", 0.3)

        gamma_cfg = com.get("gamma", {})
        gamma_limit = gamma_cfg.get("gamma_limit", [80, 120])
        gamma_p = gamma_cfg.get("probability", 0.4)

        transforms = [
            A.Rotate(limit=rot_limit, p=0.7),
        ]

        if hflip_en:
            transforms.append(A.HorizontalFlip(p=hflip_p))
        if vflip_en:
            transforms.append(A.VerticalFlip(p=vflip_p))

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

        if zoom_enabled:
            transforms.append(
                A.RandomScale(scale_limit=zoom_scale, p=zoom_p)
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
