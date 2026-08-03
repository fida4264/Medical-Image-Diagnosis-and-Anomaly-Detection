"""
tests/test_skin.py
===================
Comprehensive unit tests for Module 12: Skin Lesion Cancer Classification Framework.
Mirrors the exact structure of test_brain_mri.py, test_chest_xray.py, test_histopathology.py, test_retina.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Config Loading
# ---------------------------------------------------------------------------
class TestSkinLesionConfig:
    def test_default_config_loads(self):
        from skin.utils.config import SkinLesionConfig
        cfg = SkinLesionConfig.load()
        assert cfg.model.backbone == "efficientnet_b0"
        assert cfg.training.epochs > 0
        assert cfg.data.target_size == [224, 224]
        assert cfg.reliability.high_threshold == 0.85

    def test_config_missing_file_uses_defaults(self):
        from skin.utils.config import SkinLesionConfig
        cfg = SkinLesionConfig.load("/nonexistent/path/skin_lesion.yaml")
        assert cfg.model.dropout_rate > 0
        assert cfg.training.batch_size > 0

    def test_split_ratios_sum_to_one(self):
        from skin.utils.config import SkinLesionConfig
        cfg = SkinLesionConfig.load()
        ratios = cfg.data.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Dataset Loader — strict modality filter & HAM10000 join
# ---------------------------------------------------------------------------
class TestSkinLesionDatasetLoader:
    def _make_df(self, num_classes: int = 3, rows_per_class: int = 10) -> pd.DataFrame:
        classes = ["melanocytic_nevus", "melanoma", "benign_keratosis", "basal_cell_carcinoma",
                   "actinic_keratosis", "vascular_lesion", "dermatofibroma"][:num_classes]
        rows = []
        for c in classes:
            for j in range(rows_per_class):
                rows.append({
                    "modality": "skin",
                    "normalized_class": c,
                    "internal_id": f"{c}_{j}",
                    "is_valid": True,
                    "split": "unspecified",
                })
        # non-skin rows — must be filtered out
        rows.append({"modality": "brain_mri", "normalized_class": "glioma",
                     "internal_id": "brain_0", "is_valid": True, "split": "train"})
        rows.append({"modality": "retina", "normalized_class": "normal",
                     "internal_id": "retina_0", "is_valid": True, "split": "train"})
        return pd.DataFrame(rows)

    def test_strict_modality_filter(self, tmp_path):
        from skin.datasets.dataset_loader import SkinLesionDatasetLoader
        df = self._make_df()
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = SkinLesionDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        filtered = loader.load_and_filter()
        assert (filtered["modality"] == "skin").all()
        assert len(filtered) == 30  # 3 classes × 10 rows

    def test_dynamic_class_discovery(self, tmp_path):
        from skin.datasets.dataset_loader import SkinLesionDatasetLoader
        df = self._make_df(num_classes=5)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = SkinLesionDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        loader.load_and_filter()
        assert len(loader.label_map) == 5
        assert all(isinstance(v, int) for v in loader.label_map.values())

    def test_prepare_splits_creates_files(self, tmp_path):
        from skin.datasets.dataset_loader import SkinLesionDatasetLoader
        df = self._make_df(num_classes=3, rows_per_class=30)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        splits_path = tmp_path / "splits.csv"
        label_map_path = tmp_path / "label_map.json"

        loader = SkinLesionDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=splits_path,
            label_map_output_path=label_map_path,
        )
        loader.load_and_filter()
        splits_df = loader.prepare_splits()

        assert splits_path.is_file()
        assert label_map_path.is_file()
        assert "skin_split" in splits_df.columns
        assert set(splits_df["skin_split"].unique()) == {"train", "val", "test"}

        with open(label_map_path) as f:
            meta = json.load(f)
        assert meta["modality"] == "skin"
        assert meta["num_classes"] == 3


# ---------------------------------------------------------------------------
# 3. Model — build and forward pass
# ---------------------------------------------------------------------------
class TestSkinLesionModel:
    def test_model_builds_and_forward_pass(self):
        from skin.models.skin_model import build_skin_model
        from skin.utils.config import ModelConfig

        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False, dropout_rate=0.3)
        model = build_skin_model(cfg, num_classes=7)
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 7)

    def test_num_classes_matches_output(self):
        from skin.models.skin_model import build_skin_model
        from skin.utils.config import ModelConfig

        for n in [2, 5, 7]:
            cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
            model = build_skin_model(cfg, num_classes=n)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            assert out.shape[1] == n


# ---------------------------------------------------------------------------
# 4. PredictionReliabilityScorer
# ---------------------------------------------------------------------------
class TestPredictionReliabilityScorer:
    def test_high_reliability(self):
        from skin.inference.predict_skin import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.92)
        assert reliability == "high"
        assert review is False

    def test_moderate_reliability(self):
        from skin.inference.predict_skin import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.75)
        assert reliability == "moderate"
        assert review is False

    def test_low_reliability_triggers_manual_review(self):
        from skin.inference.predict_skin import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.50)
        assert reliability == "low"
        assert review is True


# ---------------------------------------------------------------------------
# 5. Predictor — output schema and reliability integration
# ---------------------------------------------------------------------------
class TestSkinLesionPredictor:
    def test_predictor_output_schema(self):
        from PIL import Image
        from skin.inference.predict_skin import SkinLesionPredictor
        from skin.models.skin_model import build_skin_model
        from skin.utils.config import ModelConfig

        label_map = {"melanoma": 0, "nevus": 1, "bcc": 2, "bkl": 3, "akiec": 4, "vasc": 5, "df": 6}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_skin_model(cfg, num_classes=7)

        predictor = SkinLesionPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224), color=(180, 120, 80))
        result = predictor.predict_image(dummy_img)

        assert result.predicted_class in label_map
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top_k_predictions) <= 3
        assert abs(sum(result.probability_distribution.values()) - 1.0) < 1e-4
        assert result.inference_time_ms > 0
        assert result.prediction_reliability in ["high", "moderate", "low"]
        assert isinstance(result.requires_manual_review, bool)

    def test_to_standard_dict_schema(self):
        from PIL import Image
        from skin.inference.predict_skin import SkinLesionPredictor
        from skin.models.skin_model import build_skin_model
        from skin.utils.config import ModelConfig

        label_map = {"melanoma": 0, "nevus": 1, "bcc": 2}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_skin_model(cfg, num_classes=3)

        predictor = SkinLesionPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224))
        result = predictor.predict_image(dummy_img)
        d = result.to_standard_dict()

        required_keys = {
            "module", "modality", "predicted_class", "confidence",
            "probability_distribution", "top3_predictions",
            "prediction_reliability", "requires_manual_review",
            "model_version", "inference_time_ms", "timestamp", "status",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["module"] == "skin"
        assert d["modality"] == "skin"
        assert d["status"] == "success"
        assert d["prediction_reliability"] in ["high", "moderate", "low"]
        assert isinstance(d["requires_manual_review"], bool)


# ---------------------------------------------------------------------------
# 6. CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManager:
    def test_save_and_load_best(self, tmp_path):
        from skin.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}, "val_accuracy": 0.92}
        manager.save_best(state)
        assert manager.has_best()
        loaded = manager.load_best()
        assert loaded is not None
        assert loaded["val_accuracy"] == pytest.approx(0.92)

    def test_save_and_load_last(self, tmp_path):
        from skin.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"epoch": 7, "model_state_dict": {}}
        manager.save_last(state)
        assert manager.has_last()
        loaded = manager.load_last()
        assert loaded["epoch"] == 7


# ---------------------------------------------------------------------------
# 7. Package __init__ public API
# ---------------------------------------------------------------------------
class TestSkinLesionPackageInit:
    def test_public_api_importable(self):
        import skin
        expected = [
            "SkinLesionDatasetLoader",
            "SkinLesionDataset",
            "create_skin_data_loaders",
            "SkinLesionClassifier",
            "build_skin_model",
            "SkinLesionTrainer",
            "SkinLesionEvaluator",
            "SkinLesionPredictor",
            "SkinPredictionResult",
            "PredictionReliabilityScorer",
            "ClinicalReportGenerator",
            "SkinLesionConfig",
        ]
        for name in expected:
            assert hasattr(skin, name), f"Missing from skin.__init__: {name}"

    def test_version_string(self):
        import skin
        assert skin.__version__ == "0.12.0"
