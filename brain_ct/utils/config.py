"""
brain_ct/utils/config.py
=========================
Configuration loader and dataclass for Module 9: Brain CT Disease Analysis.
Mirrors brain_mri/utils/config.py and chest_xray/utils/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "brain_ct.yaml"


@dataclass
class BrainCTDataConfig:
    metadata_path: str = "metadata/master_metadata.csv"
    processed_images_dir: str = "processed_images"
    modality_filter: str = "brain_ct"
    splits_output_path: str = "metadata/brain_ct_splits.csv"
    label_map_output_path: str = "metadata/brain_ct_label_map.json"
    target_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class BrainCTModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.4
    use_batch_norm: bool = True
    saved_model_dir: str = "saved_models/brain_ct"
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"

    @property
    def best_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.best_model_name

    @property
    def last_model_path(self) -> Path:
        return Path(self.saved_model_dir) / self.last_model_name


@dataclass
class BrainCTTrainingConfig:
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
    tensorboard_log_dir: str = "saved_models/brain_ct/tensorboard_logs"


@dataclass
class BrainCTEvaluationConfig:
    output_dir: str = "saved_models/brain_ct/evaluation"
    classification_report_json: str = "saved_models/brain_ct/classification_report.json"
    confusion_matrix_png: str = "saved_models/brain_ct/confusion_matrix.png"
    roc_curves_dir: str = "saved_models/brain_ct/roc_curves"
    misclassified_dir: str = "saved_models/brain_ct/evaluation/misclassified"
    performance_report_md: str = "saved_models/brain_ct/evaluation/performance_report.md"


@dataclass
class BrainCTExportConfig:
    onnx_path: str = "saved_models/brain_ct/best_model.onnx"
    torchscript_path: str = "saved_models/brain_ct/best_model.torchscript"
    opset_version: int = 17


@dataclass
class BrainCTInferenceConfig:
    reports_dir: str = "saved_models/brain_ct/reports"
    model_version: str = "0.9.0"


@dataclass
class BrainCTConfig:
    data: BrainCTDataConfig = field(default_factory=BrainCTDataConfig)
    model: BrainCTModelConfig = field(default_factory=BrainCTModelConfig)
    training: BrainCTTrainingConfig = field(default_factory=BrainCTTrainingConfig)
    evaluation: BrainCTEvaluationConfig = field(default_factory=BrainCTEvaluationConfig)
    export: BrainCTExportConfig = field(default_factory=BrainCTExportConfig)
    inference: BrainCTInferenceConfig = field(default_factory=BrainCTInferenceConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> BrainCTConfig:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_file():
            logger.warning(f"Config not found at {path}. Using defaults.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data_cfg = BrainCTDataConfig(**raw.get("data", {}))
        if isinstance(data_cfg.target_size, list):
            data_cfg.target_size = tuple(data_cfg.target_size)

        return cls(
            data=data_cfg,
            model=BrainCTModelConfig(**raw.get("model", {})),
            training=BrainCTTrainingConfig(**raw.get("training", {})),
            evaluation=BrainCTEvaluationConfig(**raw.get("evaluation", {})),
            export=BrainCTExportConfig(**raw.get("export", {})),
            inference=BrainCTInferenceConfig(**raw.get("inference", {})),
        )
