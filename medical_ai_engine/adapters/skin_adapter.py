"""
Medical AI Engine — Skin Lesion Predictor Adapter
Wraps Module 12 (Skin Lesion Cancer Classification Framework).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from PIL import Image

from skin.inference.predict_skin import SkinLesionPredictor
from skin.models.skin_model import build_skin_model
from skin.utils.config import SkinLesionConfig
from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter


class SkinPredictorAdapter(BasePredictorAdapter):
    """
    Adapter wrapping SkinLesionPredictor (Module 12).
    """

    @property
    def modality(self) -> str:
        return "skin"

    def load_model(self, checkpoint_path: Optional[str] = None, label_map_path: Optional[str] = None) -> None:
        ckpt = checkpoint_path or self._checkpoint_path or "saved_models/skin/best_model.pt"
        lbl_map = label_map_path or self._label_map_path or "metadata/skin_label_map.json"

        lbl_map_file = Path(lbl_map)
        if lbl_map_file.is_file():
            with open(lbl_map_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            label_map = meta.get("class_to_label", {
                "melanoma": 0, "melanocytic_nevus": 1, "basal_cell_carcinoma": 2,
                "actinic_keratosis": 3, "benign_keratosis": 4, "dermatofibroma": 5, "vascular_lesion": 6
            })
        else:
            label_map = {
                "melanoma": 0, "melanocytic_nevus": 1, "basal_cell_carcinoma": 2,
                "actinic_keratosis": 3, "benign_keratosis": 4, "dermatofibroma": 5, "vascular_lesion": 6
            }

        num_classes = len(label_map)
        cfg = SkinLesionConfig()
        model = build_skin_model(cfg.model, num_classes=num_classes)

        ckpt_file = Path(ckpt)
        if ckpt_file.is_file():
            try:
                state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                elif isinstance(state, dict):
                    model.load_state_dict(state)
            except Exception:
                pass

        self._predictor = SkinLesionPredictor(model=model, label_map=label_map)
        self._checkpoint_path = str(ckpt)
        self._label_map_path = str(lbl_map)
        self._is_loaded = True

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()
        assert self._predictor is not None

        res = self._predictor.predict_image(image)
        res_dict = res.to_standard_dict() if hasattr(res, "to_standard_dict") else res

        confidence = float(res_dict.get("confidence", 0.0))
        top3_raw = res_dict.get("top_3_predictions", res_dict.get("top3_predictions", []))
        top_k = []
        for item in top3_raw:
            if isinstance(item, (list, tuple)):
                top_k.append({"disease": item[0], "confidence": float(item[1])})
            elif isinstance(item, dict):
                top_k.append({"disease": item.get("disease", "unknown"), "confidence": float(item.get("confidence", 0.0))})

        reliability = res_dict.get("prediction_reliability") or ("high" if confidence >= 0.85 else ("moderate" if confidence >= 0.65 else "low"))
        manual_review = res_dict.get("requires_manual_review") if "requires_manual_review" in res_dict else (confidence < 0.85)

        res_dict.update({
            "predicted_class": res_dict.get("predicted_class", "unknown"),
            "confidence": confidence,
            "top_k_predictions": top_k,
            "prediction_reliability": reliability,
            "requires_manual_review": manual_review,
        })
        return res_dict

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        if self._predictor and hasattr(self._predictor, "label_map"):
            meta["supported_classes"] = list(self._predictor.label_map.keys())
        else:
            meta["supported_classes"] = [
                "melanoma", "melanocytic_nevus", "basal_cell_carcinoma",
                "actinic_keratosis", "benign_keratosis", "dermatofibroma", "vascular_lesion"
            ]
        meta["extra"] = {"prediction_reliability": True, "requires_manual_review": True}
        return meta
