"""
chest_xray/inference/predict_chest_xray.py
============================================
Single-image and batch inference for Chest X-ray Disease Classification.
Returns standardized prediction JSON compatible with Module 14 Unified Prediction Engine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from chest_xray.datasets.dataset_builder import get_chest_xray_transforms
from chest_xray.utils.config import ChestXrayInferenceConfig

logger = logging.getLogger(__name__)

MODULE_VERSION = "0.8.0"


@dataclass
class Top3Prediction:
    disease: str
    confidence: float
    rank: int


@dataclass
class ChestXrayPredictionResult:
    """Structured result from a single chest_xray inference pass."""
    predicted_class: str
    confidence: float
    top3_predictions: List[Top3Prediction]
    probabilities: Dict[str, float]
    model_version: str = MODULE_VERSION
    inference_time_ms: float = 0.0
    timestamp: str = ""
    image_path: str = ""
    status: str = "success"
    error_message: str = ""

    def to_standard_dict(self) -> Dict[str, Any]:
        """
        Returns the standardized Module 8 prediction JSON.
        This format is required by Module 14 (Unified Prediction Engine).
        """
        return {
            "module": "chest_xray",
            "modality": "chest_xray",
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "top3_predictions": [
                {
                    "rank": t.rank,
                    "disease": t.disease,
                    "confidence": round(t.confidence, 6),
                }
                for t in self.top3_predictions
            ],
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "status": self.status,
            "error_message": self.error_message,
        }


class ChestXrayPredictor:
    """Production-ready inference engine for Chest X-ray Disease Classification."""

    def __init__(
        self,
        model: nn.Module,
        label_map: Dict[str, int],
        target_size: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
        model_version: str = MODULE_VERSION,
    ):
        self.model = model
        self.label_map = label_map
        self.inv_label_map = {idx: name for name, idx in label_map.items()}
        self.target_size = target_size
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        self.transform = get_chest_xray_transforms(target_size=target_size, is_train=False)
        self.model_version = model_version

    def predict_image(self, image: Image.Image | str | Path) -> ChestXrayPredictionResult:
        """Runs inference on a single PIL image or file path."""
        image_path = ""
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1).squeeze(0)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        probs = probs_tensor.cpu().numpy().astype(float)
        pred_idx = int(np.argmax(probs))
        pred_class = self.inv_label_map.get(pred_idx, f"Class_{pred_idx}")

        prob_dict = {self.inv_label_map.get(i, f"Class_{i}"): round(float(p), 6) for i, p in enumerate(probs)}

        top3_idxs = np.argsort(probs)[::-1][:3]
        top3 = [
            Top3Prediction(
                disease=self.inv_label_map.get(int(i), f"Class_{i}"),
                confidence=round(float(probs[i]), 6),
                rank=rank + 1,
            )
            for rank, i in enumerate(top3_idxs)
        ]

        return ChestXrayPredictionResult(
            predicted_class=pred_class,
            confidence=round(float(probs[pred_idx]), 6),
            top3_predictions=top3,
            probabilities=prob_dict,
            model_version=self.model_version,
            inference_time_ms=round(inference_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_path=image_path,
            status="success",
        )

    def predict_and_save(
        self,
        image: Image.Image | str | Path,
        output_dir: str | Path = "saved_models/chest_xray/reports",
    ) -> Tuple[ChestXrayPredictionResult, Path]:
        result = self.predict_image(image)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "chest_xray_prediction.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result.to_standard_dict(), f, indent=2)
        return result, report_path
