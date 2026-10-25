"""
training/v2_baseline/dataset.py
===============================
Universal PyTorch Dataset for V2.0 manifests and split files.
Ensures zero contamination from quarantined records, masks, or object detection sets.
Handles both multiclass integer labels and multi-label binary vectors.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class V2ManifestDataset(Dataset):
    """
    Universal dataset loader for V2.0 split manifests.
    Guarantees:
      1. No quarantine records are loaded.
      2. No masks or non-classification datasets are loaded.
      3. Images are converted to RGB tensors with deterministic eval transforms.
      4. Targets match the specified task (multiclass integer or multilabel float tensor).
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        split: Optional[str] = None,
        fold: Optional[int] = None,
        is_train_fold: bool = True,
        transform: Optional[Callable] = None,
        task: str = "multiclass",
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.transform = transform
        self.task = task.lower()
        self.num_classes = num_classes

        # Load DataFrame
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            raise TypeError(f"Expected DataFrame or path, got {type(manifest)}")

        # Defensive sanity checks against quarantine/masks/VinBigData
        if "quarantine_reason" in df.columns:
            raise ValueError("Forbidden: Quarantine manifest passed directly to active training dataset!")
        if "dataset_name" in df.columns:
            forbidden_datasets = [
                "VinBigData Chest X-ray Abnormalities Detection",
                "ODIR-5K Preprocessed",
            ]
            for ds in forbidden_datasets:
                if (df["dataset_name"] == ds).any():
                    raise ValueError(f"Forbidden dataset '{ds}' detected in active manifest!")

        # Apply split / fold filtering
        if fold is not None and "fold" in df.columns:
            # Manifest has pre-labeled folds and train/validation splits per fold
            target_split = split if split is not None else ("train" if is_train_fold else "validation")
            if "split" in df.columns:
                df = df[(df["fold"] == fold) & (df["split"] == target_split)].reset_index(drop=True)
            else:
                if is_train_fold:
                    df = df[df["fold"] != fold].reset_index(drop=True)
                else:
                    df = df[df["fold"] == fold].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"No records found for fold={fold}, split={target_split}.")

        elif split is not None:
            if "split" not in df.columns:
                raise ValueError(f"Requested split='{split}' but column 'split' not in manifest.")
            df = df[df["split"] == split].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"No records found for split='{split}'.")

        self.df = df
        logger.info(
            f"V2ManifestDataset initialized: {len(self.df)} samples "
            f"(split={split}, fold={fold}, is_train_fold={is_train_fold}, task={self.task})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # 1. Load image safely
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image at {img_path}: {e}")

        # 2. Apply transform
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            import torchvision.transforms as T
            img_tensor = T.ToTensor()(img)

        # 3. Format target
        if self.task == "multilabel":
            # Target is an 8-element float tensor (for Retina BCEWithLogitsLoss)
            vec_raw = row["multi_label_vector"]
            if isinstance(vec_raw, str):
                vec = ast.literal_eval(vec_raw)
            elif isinstance(vec_raw, (list, tuple, np.ndarray)):
                vec = list(vec_raw)
            else:
                raise ValueError(f"Invalid multi_label_vector format: {vec_raw}")
            target = torch.tensor(vec, dtype=torch.float32)
        else:
            # Binary or multiclass target is a single long integer
            target = torch.tensor(int(row["class_id"]), dtype=torch.long)

        # 4. Metadata
        meta = {
            "image_id": str(row.get("image_id", idx)),
            "image_path": str(img_path),
            "standardized_label": str(row.get("standardized_label", "")),
        }

        return img_tensor, target, meta

    def get_labels(self) -> np.ndarray:
        """Returns array of labels (class_id integers for multiclass, or 2D array for multilabel)."""
        if self.task == "multilabel":
            vectors = []
            for vec_raw in self.df["multi_label_vector"]:
                if isinstance(vec_raw, str):
                    vectors.append(ast.literal_eval(vec_raw))
                else:
                    vectors.append(list(vec_raw))
            return np.array(vectors, dtype=np.float32)
        else:
            return self.df["class_id"].to_numpy(dtype=np.int64)
