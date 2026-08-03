"""
modality_classifier/config.py
==============================
Configuration loader and dataclass definition for Module 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "modality_classifier.yaml"


@dataclass
class DataConfig:
    metadata_path: str = "metadata/master_metadata.csv"
    processed_images_dir: str = "processed_images"
    splits_output_path: str = "metadata/modality_splits.csv"
    label_map_output_path: str = "metadata/modality_label_map.json"
    target_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15})
    random_seed: int = 42
    num_workers: int = 4


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    saved_model_path: str = "outputs/models/modality_classifier_best.pt"


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    loss_function: str = "cross_entropy"
    scheduler: str = "reduce_on_plateau"
    patience: int = 3
    grad_clip: float = 1.0


@dataclass
class EvaluationConfig:
    report_json_path: str = "outputs/modality_evaluation_report.json"
    report_md_path: str = "outputs/modality_evaluation_report.md"
    confusion_matrix_png: str = "outputs/modality_confusion_matrix.png"


@dataclass
class OnnxConfig:
    export_path: str = "outputs/models/modality_classifier.onnx"
    opset_version: int = 17
    input_name: str = "input_images"
    output_name: str = "modality_logits"
    dynamic_axes: Dict[str, Dict[int, str]] = field(
        default_factory=lambda: {
            "input_images": {0: "batch_size"},
            "modality_logits": {0: "batch_size"},
        }
    )


@dataclass
class ModalityClassifierConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    onnx: OnnxConfig = field(default_factory=OnnxConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> ModalityClassifierConfig:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_file():
            logger.warning(f"Config file not found at {path}. Using default configuration.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data_cfg = DataConfig(**raw.get("data", {}))
        # Ensure target_size is tuple
        if isinstance(data_cfg.target_size, list):
            data_cfg.target_size = tuple(data_cfg.target_size)

        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        evaluation_cfg = EvaluationConfig(**raw.get("evaluation", {}))
        onnx_cfg = OnnxConfig(**raw.get("onnx", {}))

        return cls(
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            evaluation=evaluation_cfg,
            onnx=onnx_cfg,
        )
