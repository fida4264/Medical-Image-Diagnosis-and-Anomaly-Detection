"""
augmentation/online/online_generator.py
=========================================
Online augmentation: Python iterator that yields augmented image batches
for use during model training without saving images to disk.

Both modes (offline and online) use the same augmentation policies.
This generator is framework-agnostic — it works with TensorFlow,
PyTorch, or any training loop that accepts numpy arrays.

Usage — Basic iteration
-----------------------
    gen = OnlineGenerator(config, factory, metadata_df)
    for batch_images, batch_labels in gen:
        # batch_images: np.ndarray shape (batch_size, H, W, 3) float32 [0,1]
        # batch_labels: np.ndarray shape (batch_size,) int
        train_step(batch_images, batch_labels)

Usage — TensorFlow integration
-------------------------------
    import tensorflow as tf
    dataset = tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        )
    )

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from augmentation.factory.augmentation_factory import AugmentationFactory

logger = logging.getLogger(__name__)


class OnlineGenerator:
    """
    Online augmentation data generator.

    Yields (batch_images, batch_labels) tuples during training.
    Images are augmented on-the-fly; nothing is saved to disk.

    Parameters
    ----------
    config : dict
        Full augmentation configuration.
    factory : AugmentationFactory
        Factory for creating modality-specific augmenters.
    metadata_df : pd.DataFrame
        DataFrame with at minimum: processed_path, modality,
        normalized_class columns.
    batch_size : int
        Number of images per batch (default: 32).
    shuffle : bool
        Whether to shuffle at each epoch (default: True).
    augment : bool
        Whether to apply augmentation (default: True).
        Set to False for validation/testing.
    seed : int
        Random seed for reproducibility.
    image_size : tuple of int
        Target (height, width) for output images (default: (224, 224)).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        factory: AugmentationFactory,
        metadata_df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = True,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self._config = config
        self._factory = factory
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._augment = augment
        self._seed = seed
        self._image_size = image_size
        self._rng = random.Random(seed)

        # Determine path column
        path_col = "processed_path" if "processed_path" in metadata_df.columns else "image_path"
        self._path_col = path_col

        # Build label encoder
        self._df = metadata_df.copy().reset_index(drop=True)
        classes = sorted(self._df["normalized_class"].unique()) if "normalized_class" in self._df.columns else []
        self._class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
        self._idx_to_class: Dict[int, str] = {i: c for c, i in self._class_to_idx.items()}

        logger.info(
            "OnlineGenerator: %d images | %d classes | batch=%d | augment=%s",
            len(self._df), len(classes), batch_size, augment,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Total number of unique classes."""
        return len(self._class_to_idx)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Mapping from class name to integer label."""
        return self._class_to_idx

    @property
    def idx_to_class(self) -> Dict[int, str]:
        """Mapping from integer label to class name."""
        return self._idx_to_class

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return max(1, len(self._df) // self._batch_size)

    # ------------------------------------------------------------------
    # Image loading & augmentation
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load an image from disk. Returns BGR uint8 or None on error."""
        img = cv2.imread(path)
        if img is None:
            logger.warning("OnlineGenerator: cannot load %s", path)
        return img

    def _process_image(
        self,
        image: np.ndarray,
        modality: str,
        seed: int,
    ) -> np.ndarray:
        """
        Apply augmentation (if enabled) and normalize to [0, 1] float32.

        Returns
        -------
        np.ndarray
            RGB float32 image of shape (H, W, 3).
        """
        if self._augment and AugmentationFactory.is_supported(modality):
            augmenter = self._factory.get(modality)
            image, _ = augmenter.augment(image, seed=seed)

        # Resize to target
        h, w = self._image_size
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB, normalize to [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over one full epoch of batches."""
        indices = list(range(len(self._df)))
        if self._shuffle:
            self._rng.shuffle(indices)

        batch_images: List[np.ndarray] = []
        batch_labels: List[int] = []

        for i, idx in enumerate(indices):
            row = self._df.iloc[idx]
            path = str(row.get(self._path_col, ""))
            modality = str(row.get("modality", ""))
            cls = str(row.get("normalized_class", "unknown"))
            label = self._class_to_idx.get(cls, 0)

            # Load image
            image = self._load_image(path)
            if image is None:
                continue

            # Per-image seed derived from base seed + index
            aug_seed = (self._seed + idx * 13 + i * 7) % (2**31)

            # Process
            processed = self._process_image(image, modality, aug_seed)
            batch_images.append(processed)
            batch_labels.append(label)

            if len(batch_images) == self._batch_size:
                yield (
                    np.stack(batch_images, axis=0),
                    np.array(batch_labels, dtype=np.int32),
                )
                batch_images = []
                batch_labels = []

        # Yield last partial batch if non-empty
        if batch_images:
            yield (
                np.stack(batch_images, axis=0),
                np.array(batch_labels, dtype=np.int32),
            )

    def stream(self, epochs: int = 1) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Endless stream over multiple epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs to iterate (default: 1).

        Yields
        ------
        (batch_images, batch_labels)
        """
        for _ in range(epochs):
            yield from self

    def __repr__(self) -> str:
        return (
            f"OnlineGenerator("
            f"images={len(self._df)}, "
            f"classes={self.num_classes}, "
            f"batch_size={self._batch_size}, "
            f"augment={self._augment})"
        )
