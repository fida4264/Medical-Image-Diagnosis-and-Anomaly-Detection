"""
histopathology/utils/config.py
================================
Configuration loader and dataclass for Module 10: Histopathology Cancer Subtyping.
Mirrors the exact pattern of brain_mri/utils/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "histopathology.yaml"


@dataclass
class DataConfig:
    metadata_path: str = "metadata/master_metadata.csv"
    processed_images_dir: str = "processed_images"
    splits_output_path: str = "metadata/histopathology_splits.csv"
    label_map_output_path: str = "metadata/histopathology_label_map.json"
    target_size: List[int] = field(default_factory=lambda: [224, 224])
    num_workers: int = 4
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout_rate: float = 0.4
    saved_model_dir: str = "saved_models/histopathology"


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 7
    min_delta: float = 0.001
    use_amp: bool = True
    scheduler: str = "cosine"
    warmup_epochs: int = 2


@dataclass
class EvaluationConfig:
    output_dir: str = "outputs/histopathology"
    generate_roc: bool = True
    generate_confusion_matrix: bool = True
    generate_misclassified: bool = True
    run_benchmark: bool = True
    benchmark_iterations: int = 50


@dataclass
class ExportConfig:
    onnx_path: str = "saved_models/histopathology/best_model.onnx"
    torchscript_path: str = "saved_models/histopathology/best_model.torchscript"
    opset_version: int = 17


@dataclass
class InferenceConfig:
    reports_dir: str = "outputs/histopathology/reports"
    top_k: int = 3


@dataclass
class HistopathologyConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "HistopathologyConfig":
        """Load configuration from YAML file, falling back to defaults."""
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not path.is_file():
            logger.warning(f"Config file not found at '{path}'. Using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        def _merge(dataclass_instance, raw_dict: dict):
            for key, val in raw_dict.items():
                if hasattr(dataclass_instance, key):
                    setattr(dataclass_instance, key, val)
            return dataclass_instance

        cfg = cls()
        cfg.data = _merge(DataConfig(), raw.get("data", {}))
        cfg.model = _merge(ModelConfig(), raw.get("model", {}))
        cfg.training = _merge(TrainingConfig(), raw.get("training", {}))
        cfg.evaluation = _merge(EvaluationConfig(), raw.get("evaluation", {}))
        cfg.export = _merge(ExportConfig(), raw.get("export", {}))
        cfg.inference = _merge(InferenceConfig(), raw.get("inference", {}))

        logger.info(f"HistopathologyConfig loaded from '{path}'.")
        return cfg
