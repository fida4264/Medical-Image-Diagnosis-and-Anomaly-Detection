"""
ood_detection/modality_adapter.py
=================================
Inference and feature extraction adapter for V2.0 baseline models.
Wraps existing checkpoints from models/v2_baseline/ or models/v2_calibrated/.
Extracts:
  1. Raw classification logits (B, C)
  2. Latent feature representations (B, 512) from ResNet-18 layer4 / global_pool.
Never modifies model weights or architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from ood_detection.config import BASELINE_MODELS_DIR, CALIBRATED_MODELS_DIR, ROOT
from training.v2_baseline.config import MODALITY_METADATA
from training.v2_baseline.dataset import V2ManifestDataset
from training.v2_baseline.model_factory import V2BaselineModel, create_baseline_model
from training.v2_baseline.transforms import get_transforms

logger = logging.getLogger(__name__)


class ModalityAdapter:
    """
    Adapter for a single modality model.
    Provides feature extraction and logit inference without altering checkpoints.
    """

    def __init__(self, modality: str, device: str = "cpu", use_calibrated: bool = False):
        if modality not in MODALITY_METADATA:
            raise ValueError(f"Unknown modality '{modality}'. Known: {list(MODALITY_METADATA.keys())}")

        self.modality = modality
        self.meta = MODALITY_METADATA[modality]
        self.device = torch.device(device)
        self.use_calibrated = use_calibrated

        # Load temperature if calibrated
        self.temperature = 1.0
        if use_calibrated:
            temp_file = CALIBRATED_MODELS_DIR / modality / "temperature.json"
            if temp_file.exists():
                import json
                with open(temp_file, "r") as f:
                    self.temperature = json.load(f).get("temperature", 1.0)

        # Load model checkpoint
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(is_train=False)

    def _load_model(self) -> V2BaselineModel:
        """Loads model weights from baseline checkpoint."""
        ckpt_dir = CALIBRATED_MODELS_DIR / self.modality if self.use_calibrated else BASELINE_MODELS_DIR / self.modality
        ckpt_path = ckpt_dir / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "fold_0" / "best_model.pt"
        if not ckpt_path.exists():
            # Fallback to baseline
            ckpt_path = BASELINE_MODELS_DIR / self.modality / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = BASELINE_MODELS_DIR / self.modality / "fold_0" / "best_model.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {self.modality} at {ckpt_path}")

        model = create_baseline_model(
            model_name="resnet18",
            num_classes=self.meta["num_classes"],
            pretrained=False,
            task=self.meta["task"],
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        return model

    @torch.no_grad()
    def forward_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a forward pass returning (logits, features).

        Args:
            images: Batch of image tensors (B, 3, H, W).

        Returns:
            logits: (B, num_classes)
            features: (B, 512) feature embeddings
        """
        images = images.to(self.device)
        # Forward through backbone
        conv_feats = self.model.backbone(images)  # (B, 512, 7, 7)
        if len(conv_feats.shape) == 4:
            pooled = self.model.global_pool(conv_feats)
            features = torch.flatten(pooled, 1)
        else:
            features = conv_feats

        # Pass through classifier
        logits = self.model.classifier(features)
        return logits, features

    @torch.no_grad()
    def process_image(self, image: Image.Image | str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """Processes a single image file or PIL Image."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_rgb = img.convert("RGB")
        else:
            img_rgb = image.convert("RGB")

        tensor = self.transform(img_rgb).unsqueeze(0)
        logits, features = self.forward_batch(tensor)
        return logits.cpu().numpy()[0], features.cpu().numpy()[0]

    @torch.no_grad()
    def extract_dataset_predictions(
        self,
        loader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference over a DataLoader, collecting logits, features, and metadata.
        """
        all_logits = []
        all_features = []
        all_labels = []
        all_ids = []
        all_paths = []

        total = 0
        for images, labels, metas in loader:
            logits, features = self.forward_batch(images)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if isinstance(metas, dict):
                all_ids.extend(metas.get("image_id", []))
                all_paths.extend(metas.get("image_path", []))

            total += len(images)
            if max_samples and total >= max_samples:
                break

        res = {
            "logits": torch.cat(all_logits).numpy(),
            "features": torch.cat(all_features).numpy(),
            "labels": torch.cat(all_labels).numpy(),
            "image_ids": all_ids,
            "image_paths": all_paths,
            "modality": self.modality,
            "task": self.meta["task"],
            "temperature": self.temperature,
        }
        return res


# Alias for explicit package naming
OODModalityAdapter = ModalityAdapter
