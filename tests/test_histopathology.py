"""
tests/test_histopathology.py
==============================
Unit tests for Module 10: Histopathology Cancer Subtyping Framework.
Mirrors the exact structure of test_brain_mri.py and test_chest_xray.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Config Loading
# ---------------------------------------------------------------------------
class TestHistopathologyConfig:
    def test_default_config_loads(self):
        from histopathology.utils.config import HistopathologyConfig

        cfg = HistopathologyConfig.load()
        assert cfg.model.backbone == "efficientnet_b0"
        assert cfg.training.epochs > 0
        assert cfg.data.target_size == [224, 224]

    def test_config_load_missing_file_uses_defaults(self):
        from histopathology.utils.config import HistopathologyConfig

        cfg = HistopathologyConfig.load("/nonexistent/path/to/config.yaml")
        assert cfg.model.dropout_rate > 0
        assert cfg.training.batch_size > 0


# ---------------------------------------------------------------------------
# 2. Dataset Loader — strict modality filter
# ---------------------------------------------------------------------------
class TestHistopathologyDatasetLoader:
    def _make_df(self, num_classes: int = 3, rows_per_class: int = 5) -> pd.DataFrame:
        classes = [f"class_{i}" for i in range(num_classes)]
        rows = []
        for c in classes:
            for j in range(rows_per_class):
                rows.append({
                    "modality": "histopathology",
                    "normalized_class": c,
                    "internal_id": f"{c}_{j}",
                    "is_valid": True,
                    "split": "unspecified",
                })
        # Add a non-histopathology row that must be filtered out
        rows.append({"modality": "brain_mri", "normalized_class": "glioma",
                     "internal_id": "brain_0", "is_valid": True, "split": "train"})
        return pd.DataFrame(rows)

    def test_strict_modality_filter(self, tmp_path):
        from histopathology.datasets.dataset_loader import HistopathologyDatasetLoader

        df = self._make_df()
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = HistopathologyDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        filtered = loader.load_and_filter()
        # Must contain only histopathology rows
        assert (filtered["modality"] == "histopathology").all()
        assert len(filtered) == 15  # 3 classes × 5 rows

    def test_dynamic_class_discovery(self, tmp_path):
        from histopathology.datasets.dataset_loader import HistopathologyDatasetLoader

        df = self._make_df(num_classes=5)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = HistopathologyDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        loader.load_and_filter()
        assert len(loader.label_map) == 5
        assert all(isinstance(v, int) for v in loader.label_map.values())

    def test_prepare_splits_creates_files(self, tmp_path):
        from histopathology.datasets.dataset_loader import HistopathologyDatasetLoader

        df = self._make_df(num_classes=3, rows_per_class=20)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        splits_path = tmp_path / "splits.csv"
        label_map_path = tmp_path / "label_map.json"

        loader = HistopathologyDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=splits_path,
            label_map_output_path=label_map_path,
        )
        loader.load_and_filter()
        splits_df = loader.prepare_splits()

        assert splits_path.is_file()
        assert label_map_path.is_file()
        assert "histo_split" in splits_df.columns
        assert set(splits_df["histo_split"].unique()) == {"train", "val", "test"}

        with open(label_map_path) as f:
            meta = json.load(f)
        assert meta["modality"] == "histopathology"
        assert meta["num_classes"] == 3


# ---------------------------------------------------------------------------
# 3. Model — build and forward pass
# ---------------------------------------------------------------------------
class TestHistopathologyModel:
    def test_model_builds_and_forward_pass(self):
        from histopathology.models.histopathology_model import build_histopathology_model
        from histopathology.utils.config import ModelConfig

        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False, dropout_rate=0.3)
        model = build_histopathology_model(cfg, num_classes=5)
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 5)

    def test_num_classes_matches_output(self):
        from histopathology.models.histopathology_model import build_histopathology_model
        from histopathology.utils.config import ModelConfig

        for n in [2, 3, 5]:
            cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
            model = build_histopathology_model(cfg, num_classes=n)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            assert out.shape[1] == n


# ---------------------------------------------------------------------------
# 4. Predictor — output schema
# ---------------------------------------------------------------------------
class TestHistopathologyPredictor:
    def test_predictor_output_schema(self):
        from PIL import Image
        from histopathology.inference.predict_histopathology import HistopathologyPredictor
        from histopathology.models.histopathology_model import build_histopathology_model
        from histopathology.utils.config import ModelConfig

        label_map = {
            "colon_adenocarcinoma": 0,
            "colon_normal": 1,
            "lung_adenocarcinoma": 2,
            "lung_normal": 3,
            "lung_squamous_cell": 4,
        }
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_histopathology_model(cfg, num_classes=5)

        predictor = HistopathologyPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
        result = predictor.predict_image(dummy_img)

        assert result.predicted_class in label_map
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top_k_predictions) <= 3
        assert abs(sum(result.probability_distribution.values()) - 1.0) < 1e-4
        assert result.inference_time_ms > 0

    def test_to_standard_dict_schema(self):
        from PIL import Image
        from histopathology.inference.predict_histopathology import HistopathologyPredictor
        from histopathology.models.histopathology_model import build_histopathology_model
        from histopathology.utils.config import ModelConfig

        label_map = {"colon_adenocarcinoma": 0, "colon_normal": 1, "lung_adenocarcinoma": 2}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_histopathology_model(cfg, num_classes=3)

        predictor = HistopathologyPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224))
        result = predictor.predict_image(dummy_img)
        d = result.to_standard_dict()

        required_keys = {
            "module", "modality", "predicted_class", "confidence",
            "probability_distribution", "top_3_predictions",
            "model_version", "inference_time_ms", "prediction_timestamp",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["module"] == "histopathology"
        assert d["modality"] == "histopathology"


# ---------------------------------------------------------------------------
# 5. Config — split ratios sum to 1.0
# ---------------------------------------------------------------------------
class TestHistopathologyConfigSplitRatios:
    def test_split_ratios_sum_to_one(self):
        from histopathology.utils.config import HistopathologyConfig

        cfg = HistopathologyConfig.load()
        ratios = cfg.data.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 6. CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManager:
    def test_save_and_load_best(self, tmp_path):
        from histopathology.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(save_dir=tmp_path)
        state = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}, "val_accuracy": 0.95}
        manager.save_best(state)
        assert manager.has_best()

        loaded = manager.load_best()
        assert loaded is not None
        assert loaded["val_accuracy"] == pytest.approx(0.95)

    def test_save_and_load_last(self, tmp_path):
        from histopathology.utils.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(save_dir=tmp_path)
        state = {"epoch": 5, "model_state_dict": {}}
        manager.save_last(state)
        assert manager.has_last()

        loaded = manager.load_last()
        assert loaded["epoch"] == 5


# ---------------------------------------------------------------------------
# 7. Package __init__ public API
# ---------------------------------------------------------------------------
class TestHistopathologyPackageInit:
    def test_public_api_importable(self):
        import histopathology
        expected = [
            "HistopathologyDatasetLoader",
            "HistopathologyDataset",
            "create_histopathology_data_loaders",
            "HistopathologyClassifier",
            "build_histopathology_model",
            "HistopathologyTrainer",
            "HistopathologyEvaluator",
            "HistopathologyPredictor",
            "HistopathologyPredictionResult",
            "ClinicalReportGenerator",
            "HistopathologyConfig",
        ]
        for name in expected:
            assert hasattr(histopathology, name), f"Missing from histopathology.__init__: {name}"

    def test_version_string(self):
        import histopathology
        assert histopathology.__version__ == "0.10.0"
