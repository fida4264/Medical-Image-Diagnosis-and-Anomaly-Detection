"""
brain_mri/utils/config.py
===========================
Configuration loader and dataclass for Module 7: Brain MRI Disease Analysis.

Mirrors the exact pattern of modality_classifier/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "brain_mri.yaml"


@dataclass
class BrainMRIDataConfig:
    metadata_path: str = "metadata/master_metadata.csv"
    processed_images_dir: str = "processed_images"
    modality_filter: str = "brain_mri"
    splits_output_path: str = "metadata/brain_mri_splits.csv"
    label_map_output_path: str = "metadata/brain_mri_label_map.json"
    target_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class BrainMRIModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.4
    use_batch_norm: bool = True
    saved_model_dir: str = "saved_models/brain_mri"
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"

    @property
    def best_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.best_model_name

    @property
    def last_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.last_model_name


@dataclass
class BrainMRITrainingConfig:
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
    tensorboard_log_dir: str = "saved_models/brain_mri/tensorboard_logs"


@dataclass
class BrainMRIEvaluationConfig:
    output_dir: str = "saved_models/brain_mri/evaluation"
    classification_report_json: str = "saved_models/brain_mri/classification_report.json"
    confusion_matrix_png: str = "saved_models/brain_mri/confusion_matrix.png"
    roc_curves_dir: str = "saved_models/brain_mri/roc_curves"
    misclassified_dir: str = "saved_models/brain_mri/evaluation/misclassified"
    performance_report_md: str = "saved_models/brain_mri/evaluation/performance_report.md"


@dataclass
class BrainMRIExportConfig:
    onnx_path: str = "saved_models/brain_mri/best_model.onnx"
    torchscript_path: str = "saved_models/brain_mri/best_model.torchscript"
    opset_version: int = 17


@dataclass
class BrainMRIInferenceConfig:
    reports_dir: str = "saved_models/brain_mri/reports"
    model_version: str = "0.7.0"


@dataclass
class BrainMRIConfig:
    data: BrainMRIDataConfig = field(default_factory=BrainMRIDataConfig)
    model: BrainMRIModelConfig = field(default_factory=BrainMRIModelConfig)
    training: BrainMRITrainingConfig = field(default_factory=BrainMRITrainingConfig)
    evaluation: BrainMRIEvaluationConfig = field(default_factory=BrainMRIEvaluationConfig)
    export: BrainMRIExportConfig = field(default_factory=BrainMRIExportConfig)
    inference: BrainMRIInferenceConfig = field(default_factory=BrainMRIInferenceConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> BrainMRIConfig:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_file():
            logger.warning(f"Config not found at {path}. Using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data_cfg = BrainMRIDataConfig(**raw.get("data", {}))
        if isinstance(data_cfg.target_size, list):
            data_cfg.target_size = tuple(data_cfg.target_size)

        return cls(
            data=data_cfg,
            model=BrainMRIModelConfig(**raw.get("model", {})),
            training=BrainMRITrainingConfig(**raw.get("training", {})),
            evaluation=BrainMRIEvaluationConfig(**raw.get("evaluation", {})),
            export=BrainMRIExportConfig(**raw.get("export", {})),
            inference=BrainMRIInferenceConfig(**raw.get("inference", {})),
        )
