"""
augmentation/offline/offline_generator.py
===========================================
Offline augmentation: loads processed images, applies policies,
and saves augmented copies to disk.

Features
--------
- Thread-safe parallel processing (ThreadPoolExecutor)
- Class imbalance detection and targeting
- Configurable augmentations per image
- Skip-existing support (resumable)
- Full metadata recording for every augmented image
- Does NOT modify original processed images

Usage
-----
    generator = OfflineGenerator(config, factory)
    results = generator.run(
        metadata_df=preprocessing_metadata_df,
        output_dir=Path("augmented_images"),
        augmentations_per_image=5,
    )

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from augmentation.factory.augmentation_factory import AugmentationFactory

logger = logging.getLogger(__name__)


class OfflineGenerator:
    """
    Generates augmented images and saves them to disk.

    Parameters
    ----------
    config : dict
        Full augmentation configuration.
    factory : AugmentationFactory
        Factory for creating modality-specific augmenters.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        factory: AugmentationFactory,
    ) -> None:
        self._config = config
        self._factory = factory

    # ------------------------------------------------------------------
    # Class imbalance analysis
    # ------------------------------------------------------------------

    def analyze_class_distribution(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze class distribution per modality.

        Returns
        -------
        dict
            {modality: {class: count}} mapping.
        """
        distribution: Dict[str, Dict[str, int]] = {}
        for modality in df["modality"].unique():
            mod_df = df[df["modality"] == modality]
            if "normalized_class" in mod_df.columns:
                counts = mod_df["normalized_class"].value_counts().to_dict()
            else:
                counts = {"unknown": len(mod_df)}
            distribution[str(modality)] = {str(k): int(v) for k, v in counts.items()}
        return distribution

    def compute_augmentation_targets(
        self,
        df: pd.DataFrame,
        augmentations_per_image: int,
        class_balancing: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Compute how many augmented images to generate per (modality, class).

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessing metadata DataFrame.
        augmentations_per_image : int
            Default number of augmentations per original image.
        class_balancing : dict
            Class balancing configuration section.

        Returns
        -------
        dict
            {f"{modality}/{normalized_class}": target_aug_count}
        """
        targets: Dict[str, int] = {}
        strategy = class_balancing.get("strategy", "minority_only")
        enabled = class_balancing.get("enabled", True)
        min_ratio = class_balancing.get("min_class_ratio", 0.5)
        target_count = class_balancing.get("target_count", None)

        dist = self.analyze_class_distribution(df)

        for modality, class_counts in dist.items():
            if not class_counts:
                continue

            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalanced = (min_count / max_count < min_ratio) if max_count > 0 else False

            for cls, count in class_counts.items():
                key = f"{modality}/{cls}"
                if not enabled:
                    targets[key] = augmentations_per_image
                elif strategy == "minority_only":
                    if imbalanced and count < max_count:
                        # Augment minority class more aggressively
                        needed = max_count - count
                        per_img = max(1, needed // max(count, 1))
                        targets[key] = min(per_img + augmentations_per_image, 20)
                    else:
                        targets[key] = augmentations_per_image
                elif strategy == "all_classes":
                    targets[key] = augmentations_per_image
                elif strategy == "target_count" and target_count is not None:
                    needed = max(0, int(target_count) - count)
                    targets[key] = max(0, needed // max(count, 1))
                else:
                    targets[key] = augmentations_per_image

        return targets

    # ------------------------------------------------------------------
    # Single-image augmentation (called from threads)
    # ------------------------------------------------------------------

    def _augment_one_image(
        self,
        row: Dict[str, Any],
        aug_index: int,
        seed: int,
        output_dir: Path,
        output_format: str,
        skip_existing: bool,
    ) -> Dict[str, Any]:
        """
        Augment one image and save it.

        Returns a result dict for the metadata writer.
        """
        original_id = str(row.get("internal_id", row.get("image_id", "")))
        original_path = str(row.get("processed_path", row.get("image_path", "")))
        modality = str(row.get("modality", ""))
        norm_class = str(row.get("normalized_class", "unknown"))

        # Determine output path
        aug_stem = f"{original_id}_aug{aug_index:03d}"
        ext = f".{output_format.lstrip('.')}"
        out_path = output_dir / modality / norm_class / f"{aug_stem}{ext}"

        if skip_existing and out_path.exists():
            return {
                "original_image_id": original_id,
                "original_path": original_path,
                "augmented_path": str(out_path),
                "modality": modality,
                "normalized_class": norm_class,
                "augmentation_index": aug_index,
                "operations_applied": [],
                "seed": seed,
                "status": "skipped",
            }

        status = "success"
        ops: List[str] = []

        try:
            # Load
            image = cv2.imread(original_path)
            if image is None:
                raise FileNotFoundError(f"Cannot load: {original_path}")

            # Get augmenter for modality
            if not AugmentationFactory.is_supported(modality):
                raise ValueError(f"Unsupported modality: {modality}")

            augmenter = self._factory.get(modality)
            aug_image, ops = augmenter.augment(image, seed=seed)

            # Save
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), aug_image)

        except FileNotFoundError as exc:
            status = "error:file_not_found"
            logger.warning("[SKIP] %s aug%d: %s", original_id, aug_index, exc)
        except ValueError as exc:
            status = "error:unsupported_modality"
            logger.warning("[SKIP] %s aug%d: %s", original_id, aug_index, exc)
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            logger.error("[ERROR] %s aug%d: %s", original_id, aug_index, exc)

        return {
            "original_image_id": original_id,
            "original_path": original_path,
            "augmented_path": str(out_path) if status == "success" else "",
            "modality": modality,
            "normalized_class": norm_class,
            "augmentation_index": aug_index,
            "operations_applied": ops,
            "seed": seed,
            "status": status,
        }

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        metadata_df: pd.DataFrame,
        output_dir: Path,
        augmentations_per_image: int = 5,
        num_workers: int = 8,
        skip_existing: bool = True,
        output_format: str = "png",
        base_seed: int = 42,
        modality_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run offline augmentation for all images in metadata_df.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Preprocessing metadata (from Module 3 or master_metadata).
        output_dir : Path
            Root directory for augmented images.
        augmentations_per_image : int
            Default augmented copies per original.
        num_workers : int
            Parallel processing threads.
        skip_existing : bool
            Skip already-augmented images.
        output_format : str
            Output image format (png or jpg).
        base_seed : int
            Base random seed for reproducibility.
        modality_filter : str or None
            If set, only augment this modality.

        Returns
        -------
        list of dict
            Result dicts for every augmentation attempt.
        """
        if modality_filter:
            metadata_df = metadata_df[metadata_df["modality"] == modality_filter]
            logger.info("Filtered to modality '%s': %d images", modality_filter, len(metadata_df))

        # Determine path column
        path_col = "processed_path" if "processed_path" in metadata_df.columns else "image_path"

        # Filter to valid rows only
        valid_df = metadata_df[metadata_df[path_col].notna()].copy()
        if "status" in valid_df.columns:
            valid_df = valid_df[valid_df["status"].isin(["success", "skipped"])]

        class_balancing = self._config.get("class_balancing", {})
        targets = self.compute_augmentation_targets(
            valid_df, augmentations_per_image, class_balancing
        )

        # Build job list
        jobs: List[Tuple[Dict, int, int]] = []
        rows = valid_df.to_dict("records")
        for row in rows:
            modality = str(row.get("modality", ""))
            norm_class = str(row.get("normalized_class", "unknown"))
            key = f"{modality}/{norm_class}"
            n_aug = targets.get(key, augmentations_per_image)
            original_id = str(row.get("internal_id", row.get("image_id", "")))
            for aug_idx in range(n_aug):
                # Deterministic seed: base + hash of id + aug index
                seed = (base_seed + hash(original_id) + aug_idx * 31) % (2**31)
                jobs.append((row, aug_idx, seed))

        logger.info(
            "Offline augmentation: %d originals → %d jobs (%.1f× avg)",
            len(rows), len(jobs),
            len(jobs) / max(len(rows), 1),
        )

        all_results: List[Dict[str, Any]] = []

        class _Progress:
            def __init__(self, total: int) -> None:
                self.total = total
                self._done = 0
                self._lock = threading.Lock()
                self._t0 = time.perf_counter()

            def tick(self) -> None:
                with self._lock:
                    self._done += 1
                    if self._done % 1000 == 0 or self._done == self.total:
                        elapsed = time.perf_counter() - self._t0
                        rate = self._done / elapsed if elapsed > 0 else 0
                        pct = self._done / self.total * 100
                        print(
                            f"  Progress: {self._done:,}/{self.total:,}"
                            f" ({pct:.1f}%) | {rate:.0f} aug/s",
                            flush=True,
                        )

        progress = _Progress(len(jobs))

        def _do_job(job: Tuple[Dict, int, int]) -> Dict:
            row, aug_idx, seed = job
            result = self._augment_one_image(
                row, aug_idx, seed, output_dir, output_format, skip_existing
            )
            progress.tick()
            return result

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_do_job, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    logger.error("Unexpected future error: %s", exc)

        success = sum(1 for r in all_results if r["status"] == "success")
        skipped = sum(1 for r in all_results if r["status"] == "skipped")
        errors = len(all_results) - success - skipped
        logger.info(
            "Offline generation complete: %d success | %d skipped | %d errors",
            success, skipped, errors,
        )
        return all_results
