"""
chest_xray/utils/config.py
============================
Configuration loader and dataclass for Module 8: Chest X-ray Disease Analysis.
Mirrors the exact pattern of brain_mri/utils/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "chest_xray.yaml"


@dataclass
class ChestXrayDataConfig:
    metadata_path: str = "metadata/master_metadata.csv"
    processed_images_dir: str = "processed_images"
    modality_filter: str = "chest_xray"
    splits_output_path: str = "metadata/chest_xray_splits.csv"
    label_map_output_path: str = "metadata/chest_xray_label_map.json"
    target_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class ChestXrayModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.4
    use_batch_norm: bool = True
    saved_model_dir: str = "saved_models/chest_xray"
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"

    @property
    def best_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.best_model_name

    @property
    def last_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.last_model_name


@dataclass
class ChestXrayTrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    loss_function: str = "cross_entropy"
    scheduler: str = "reduce_on_plateau"
    patience: int = 5
    grad_clip: float = 1.0
    mixed_precision: bool = True
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "saved_models/chest_xray/tensorboard_logs"


@dataclass
class ChestXrayEvaluationConfig:
    output_dir: str = "saved_models/chest_xray/evaluation"
    classification_report_json: str = "saved_models/chest_xray/classification_report.json"
    confusion_matrix_png: str = "saved_models/chest_xray/confusion_matrix.png"
    roc_curves_dir: str = "saved_models/chest_xray/roc_curves"
    misclassified_dir: str = "saved_models/chest_xray/evaluation/misclassified"
    performance_report_md: str = "saved_models/chest_xray/evaluation/performance_report.md"


@dataclass
class ChestXrayExportConfig:
    onnx_path: str = "saved_models/chest_xray/best_model.onnx"
    torchscript_path: str = "saved_models/chest_xray/best_model.torchscript"
    opset_version: int = 17


@dataclass
class ChestXrayInferenceConfig:
    reports_dir: str = "saved_models/chest_xray/reports"
    model_version: str = "0.8.0"


@dataclass
class ChestXrayConfig:
    data: ChestXrayDataConfig = field(default_factory=ChestXrayDataConfig)
    model: ChestXrayModelConfig = field(default_factory=ChestXrayModelConfig)
    training: ChestXrayTrainingConfig = field(default_factory=ChestXrayTrainingConfig)
    evaluation: ChestXrayEvaluationConfig = field(default_factory=ChestXrayEvaluationConfig)
    export: ChestXrayExportConfig = field(default_factory=ChestXrayExportConfig)
    inference: ChestXrayInferenceConfig = field(default_factory=ChestXrayInferenceConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> ChestXrayConfig:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_file():
            logger.warning(f"Config not found at {path}. Using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data_cfg = ChestXrayDataConfig(**raw.get("data", {}))
        if isinstance(data_cfg.target_size, list):
            data_cfg.target_size = tuple(data_cfg.target_size)

        return cls(
            data=data_cfg,
            model=ChestXrayModelConfig(**raw.get("model", {})),
            training=ChestXrayTrainingConfig(**raw.get("training", {})),
            evaluation=ChestXrayEvaluationConfig(**raw.get("evaluation", {})),
            export=ChestXrayExportConfig(**raw.get("export", {})),
            inference=ChestXrayInferenceConfig(**raw.get("inference", {})),
        )
