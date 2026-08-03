"""
modality_classifier/inference/predictor.py
===========================================
Production inference prediction engine for Modality Classification & Downstream Routing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modality_classifier.datasets.data_loader import IMAGENET_MEAN, IMAGENET_STD
from modality_classifier.models.base_model import ModalityClassifier

logger = logging.getLogger(__name__)

# Downstream Module Routing Mapping
ROUTING_MAP = {
    "brain_mri": "Module 7 - Brain MRI Anomaly Detection & Segmentation",
    "brain_ct": "Module 8 - Brain CT Scan Classification",
    "chest_xray": "Module 9 - Chest X-Ray Disease Classification",
    "histopathology": "Module 10 - Histopathology Cancer Subtyping",
    "retina": "Module 11 - Retina Diabetic Retinopathy Classification",
    "skin": "Module 12 - Skin Lesion Cancer Classification",
    "ultrasound": "Module 13 - Ultrasound Breast Anomaly Detection",
}


@dataclass
class ModalityPredictionResult:
    predicted_modality: str
    confidence: float
    top_k_predictions: List[Dict[str, Any]]
    inference_time_ms: float
    routing_destination: str
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModalityPredictor:
    """
    Inference predictor for single-image and batch modality identification.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "outputs/models/modality_classifier_best.pt",
        label_map_path: str | Path = "metadata/modality_label_map.json",
        device: Optional[str] = None,
        target_size: tuple[int, int] = (224, 224),
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.label_map_path = Path(label_map_path)
        self.target_size = target_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load label map
        self.label_map, self.inv_label_map = self._load_label_map()

        # Load model checkpoint
        self.model = self._load_model()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_label_map(self) -> tuple[Dict[str, int], Dict[int, str]]:
        if not self.label_map_path.is_file():
            # Fallback default modalities
            default_mods = ["brain_ct", "brain_mri", "chest_xray", "histopathology", "retina", "skin", "ultrasound"]
            l_map = {m: i for i, m in enumerate(default_mods)}
            inv_map = {i: m for m, i in l_map.items()}
            return l_map, inv_map

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        l_map = data.get("modality_to_label", {})
        inv_map = {int(k): v for k, v in data.get("label_to_modality", {}).items()}
        return l_map, inv_map

    def _load_model(self) -> ModalityClassifier:
        num_classes = len(self.inv_label_map)
        if not self.checkpoint_path.is_file():
            logger.warning(f"Checkpoint file {self.checkpoint_path} not found. Initializing unweighted model.")
            model = ModalityClassifier(num_classes=num_classes)
            model.to(self.device)
            model.eval()
            return model

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        backbone = checkpoint.get("metadata", {}).get("backbone", "mobilenet_v3_small")

        model = ModalityClassifier(num_classes=num_classes, backbone_name=backbone)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info(f"Loaded trained ModalityClassifier model from {self.checkpoint_path}")
        return model

    def _preprocess_input(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(img).unsqueeze(0)  # Shape: (1, 3, H, W)
        return tensor

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> ModalityPredictionResult:
        """
        Runs prediction for a single image and returns ModalityPredictionResult.
        """
        t0 = time.perf_counter()

        tensor_img = self._preprocess_input(image_input).to(self.device)
        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_preds = [
            {
                "modality": self.inv_label_map.get(int(idx), f"Class_{idx}"),
                "confidence": float(probs[idx]),
            }
            for idx in top_indices
        ]

        top_pred = top_k_preds[0]
        pred_modality = top_pred["modality"]
        confidence = top_pred["confidence"]

        routing = ROUTING_MAP.get(pred_modality, f"Module 7-13 ({pred_modality} Disease Model)")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        image_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None

        return ModalityPredictionResult(
            predicted_modality=pred_modality,
            confidence=confidence,
            top_k_predictions=top_k_preds,
            inference_time_ms=round(elapsed_ms, 2),
            routing_destination=routing,
            image_path=image_path_str,
        )
