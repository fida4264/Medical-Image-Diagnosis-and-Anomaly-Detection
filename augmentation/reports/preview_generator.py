"""
augmentation/reports/preview_generator.py
===========================================
Generates augmentation_preview/ comparison grids.

For each modality: one original image → 5 random augmented versions
displayed side-by-side in a PNG grid.

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from augmentation.base.base_augmenter import BaseAugmenter

logger = logging.getLogger(__name__)


class PreviewGenerator:
    """
    Generate comparison grids showing original vs augmented images.

    Usage
    -----
        gen = PreviewGenerator(output_dir=Path("augmentation_preview"))
        gen.generate_modality_preview(
            modality="brain_mri",
            image_path=Path("processed_images/brain_mri/glioma/MRI_000001.png"),
            augmenter=augmenter,
            n_versions=5,
        )
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_modality_preview(
        self,
        modality: str,
        image_path: Path,
        augmenter: BaseAugmenter,
        n_versions: int = 5,
        seed_offset: int = 0,
        dpi: int = 120,
        figsize: tuple = (24, 6),
    ) -> Optional[Path]:
        """
        Generate a side-by-side comparison grid for one modality.

        Layout: [Original | Aug-1 | Aug-2 | Aug-3 | Aug-4 | Aug-5]

        Parameters
        ----------
        modality : str
            Modality name (used for the output filename).
        image_path : Path
            Path to the source preprocessed image.
        augmenter : BaseAugmenter
            Configured augmenter for this modality.
        n_versions : int
            Number of augmented versions to show (default: 5).
        seed_offset : int
            Starting seed for augmentation reproducibility.
        dpi : int
            Output image DPI.
        figsize : tuple
            Figure size (width, height) in inches.

        Returns
        -------
        Path or None
            Path to the saved grid image, or None on error.
        """
        # Load original
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning("Preview: source image not found: %s", image_path)
            return None

        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            logger.warning("Preview: could not load image: %s", image_path)
            return None
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # Generate augmented versions
        augmented_versions = []
        for i in range(n_versions):
            seed = seed_offset + i * 7 + 100  # deterministic seed per version
            try:
                aug_bgr, ops = augmenter.augment(orig_bgr, seed=seed)
                aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
                augmented_versions.append((aug_rgb, ops))
            except Exception as exc:
                logger.warning("Preview augmentation failed for %s: %s", modality, exc)
                augmented_versions.append((orig_rgb.copy(), []))

        # Build figure
        n_cols = 1 + n_versions
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        fig.patch.set_facecolor("#0D1117")

        title = f"{modality.replace('_', ' ').title()} — Augmentation Preview"
        fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.02)

        # Original
        ax0 = axes[0]
        ax0.imshow(orig_rgb)
        ax0.set_title("Original", color="#58A6FF", fontsize=10, fontweight="bold")
        ax0.axis("off")
        ax0.set_facecolor("#0D1117")

        # Augmented versions
        for i, (aug_img, ops) in enumerate(augmented_versions):
            ax = axes[i + 1]
            ax.imshow(aug_img)
            op_label = f"Aug {i + 1}"
            ax.set_title(op_label, color="#3FB950", fontsize=9)
            # Show first 3 ops as subtitle
            if ops:
                short_ops = ", ".join(ops[:3])
                if len(ops) > 3:
                    short_ops += f" +{len(ops)-3}"
                ax.set_xlabel(short_ops, color="#8B949E", fontsize=7)
            ax.axis("off")
            ax.set_facecolor("#0D1117")

        plt.tight_layout()

        out_path = self._output_dir / f"{modality}_preview.png"
        plt.savefig(
            str(out_path),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="#0D1117",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info("Preview saved: %s", out_path)
        return out_path

    def generate_all_modalities(
        self,
        modality_image_map: Dict[str, Path],
        augmenter_map: Dict[str, BaseAugmenter],
        n_versions: int = 5,
        dpi: int = 120,
    ) -> List[Path]:
        """
        Generate preview grids for all available modalities.

        Parameters
        ----------
        modality_image_map : dict
            Mapping of modality → source image path.
        augmenter_map : dict
            Mapping of modality → configured augmenter.
        n_versions : int
            Number of augmented versions per modality.
        dpi : int
            Output DPI.

        Returns
        -------
        list of Path
            Paths to all saved grid images.
        """
        saved: List[Path] = []
        for modality, img_path in sorted(modality_image_map.items()):
            augmenter = augmenter_map.get(modality)
            if augmenter is None:
                logger.warning("No augmenter found for modality: %s", modality)
                continue
            path = self.generate_modality_preview(
                modality=modality,
                image_path=img_path,
                augmenter=augmenter,
                n_versions=n_versions,
                dpi=dpi,
            )
            if path:
                saved.append(path)
        logger.info("Generated %d preview grids.", len(saved))
        return saved
