"""
augmentation/policies/histopathology_policy.py
================================================
Augmentation policy for Histopathology (H&E stained slide) images.

Medical Safety Rules
--------------------
- Full 360° rotation valid (no fixed tissue orientation on slides)
- Horizontal AND vertical flip valid
- Color jitter (different staining protocols produce color variation)
- Stain variation (simulate different H&E concentrations)
- Brightness/contrast (scanner illumination variation)
- Random crop (different magnification / field of view)

Author: Medical AI System — Module 4
"""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A

from augmentation.base.base_augmenter import BaseAugmenter


class HistopathologyAugmenter(BaseAugmenter):
    """Augmentation policy for Histopathology modality."""

    @property
    def modality(self) -> str:
        return "histopathology"

    def build_pipeline(self, config: Dict[str, Any]) -> A.Compose:
        """
        Build Histopathology augmentation pipeline.

        Transforms:
        - Full rotation (tissue orientation is arbitrary)
        - Horizontal + vertical flip
        - Brightness/contrast (scanner calibration variation)
        - Color jitter (staining protocol variation)
        - Hue/saturation shift (simulate different H&E concentrations)
        - Random crop (FOV / magnification variation)
        - Gaussian noise
        """
        mod = config.get("modalities", {}).get("histopathology", {})
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
        cj_hue = cj.get("hue_shift_limit", 20)
        cj_sat = cj.get("sat_shift_limit", 30)
        cj_val = cj.get("val_shift_limit", 20)
        cj_p = cj.get("probability", 0.6)

        stain_cfg = mod.get("stain_variation", {})
        stain_enabled = stain_cfg.get("enabled", True)
        stain_p = stain_cfg.get("probability", 0.4)

        crop_cfg = mod.get("crop", com.get("crop", {}))
        crop_enabled = crop_cfg.get("enabled", True)
        crop_min_max = crop_cfg.get("min_max_height", [180, 224])
        crop_p = crop_cfg.get("probability", 0.4)

        noise = mod.get("gaussian_noise", com.get("gaussian_noise", {}))
        noise_var = noise.get("var_limit", [5, 25])
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

        # Stain variation: use RGBShift to simulate different H&E concentrations
        if stain_enabled:
            transforms.append(
                A.RGBShift(
                    r_shift_limit=15,
                    g_shift_limit=10,
                    b_shift_limit=15,
                    p=stain_p,
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
