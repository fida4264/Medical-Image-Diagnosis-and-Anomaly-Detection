"""
training/v2_baseline/config.py
==============================
Dataclass-based experiment configuration system for V2.0 baseline training.
Supports single-split, group-split, and k-fold cross-validation configurations.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT / "data" / "manifests"
SPLITS_DIR = MANIFESTS_DIR / "splits"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models" / "v2_baseline"

# Canonical modality settings
MODALITY_METADATA: Dict[str, Dict[str, Any]] = {
    "brain_mri": {
        "task": "multiclass",
        "num_classes": 4,
        "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "class_id_mapping": {0: "glioma", 1: "meningioma", 2: "no_tumor", 3: "pituitary"},
        "split_file": str(SPLITS_DIR / "brain_mri_v2.0_splits.csv"),
        "split_type": "fixed",
        "default_loss": "cross_entropy",
        "use_class_weights": False,
        "recommended_epochs": 10,
        "group_column": None,
    },
    "brain_ct": {
        "task": "binary",
        "num_classes": 2,
        "class_names": ["normal", "hemorrhage"],
        "class_id_mapping": {0: "normal", 1: "hemorrhage"},
        "split_file": str(SPLITS_DIR / "brain_ct_v2.0_folds.csv"),
        "split_type": "kfold",
        "k_folds": 5,
        "default_loss": "cross_entropy",
        "use_class_weights": False,
        "recommended_epochs": 10,
        "group_column": None,
    },
    "chest_xray": {
        "task": "multiclass",
        "num_classes": 4,
        "class_names": ["covid", "lung_opacity", "normal", "viral_pneumonia"],
        "class_id_mapping": {0: "covid", 1: "lung_opacity", 2: "normal", 3: "viral_pneumonia"},
        "split_file": str(SPLITS_DIR / "chest_xray_v2.0_splits.csv"),
        "split_type": "fixed",
        "default_loss": "weighted_ce",
        "use_class_weights": True,
        "recommended_epochs": 10,
        "group_column": None,
    },
    "histopathology": {
        "task": "multiclass",
        "num_classes": 5,
        "class_names": [
            "colon_adenocarcinoma",
            "colon_normal",
            "lung_adenocarcinoma",
            "lung_normal",
            "lung_squamous_cell_carcinoma",
        ],
        "class_id_mapping": {
            0: "colon_adenocarcinoma",
            1: "colon_normal",
            2: "lung_adenocarcinoma",
            3: "lung_normal",
            4: "lung_squamous_cell_carcinoma",
        },
        "split_file": str(SPLITS_DIR / "histopathology_v2.0_splits.csv"),
        "split_type": "fixed",
        "default_loss": "cross_entropy",
        "use_class_weights": False,
        "recommended_epochs": 5,  # 20k images on CPU
        "group_column": None,
    },
    "retina": {
        "task": "multilabel",
        "num_classes": 8,
        "class_names": [
            "normal",
            "diabetic_retinopathy",
            "glaucoma",
            "cataract",
            "amd",
            "hypertension",
            "pathological_myopia",
            "other_abnormality",
        ],
        "class_id_mapping": {
            0: "normal",
            1: "diabetic_retinopathy",
            2: "glaucoma",
            3: "cataract",
            4: "amd",
            5: "hypertension",
            6: "pathological_myopia",
            7: "other_abnormality",
        },
        "split_file": str(SPLITS_DIR / "retina_v2.0_splits.csv"),
        "split_type": "fixed",
        "default_loss": "bce_with_logits",
        "use_class_weights": True,  # per-label positive weighting
        "recommended_epochs": 10,
        "group_column": "patient_id",
    },
    "skin": {
        "task": "multiclass",
        "num_classes": 7,
        "class_names": [
            "melanocytic_nevus",
            "melanoma",
            "benign_keratosis",
            "basal_cell_carcinoma",
            "actinic_keratosis",
            "vascular_lesion",
            "dermatofibroma",
        ],
        "class_id_mapping": {
            0: "melanocytic_nevus",
            1: "melanoma",
            2: "benign_keratosis",
            3: "basal_cell_carcinoma",
            4: "actinic_keratosis",
            5: "vascular_lesion",
            6: "dermatofibroma",
        },
        "split_file": str(SPLITS_DIR / "skin_v2.0_splits.csv"),
        "split_type": "fixed",
        "default_loss": "weighted_ce",
        "use_class_weights": True,
        "recommended_epochs": 10,
        "group_column": "lesion_id",
    },
    "ultrasound": {
        "task": "multiclass",
        "num_classes": 3,
        "class_names": ["benign", "malignant", "normal"],
        "class_id_mapping": {0: "benign", 1: "malignant", 2: "normal"},
        "split_file": str(SPLITS_DIR / "ultrasound_v2.0_folds.csv"),
        "split_type": "kfold",
        "k_folds": 5,
        "default_loss": "weighted_ce",
        "use_class_weights": True,
        "recommended_epochs": 10,
        "group_column": None,
    },
}


@dataclass
class ModalityConfig:
    modality: str
    task: str
    num_classes: int
    class_names: List[str]
    class_id_mapping: Dict[int, str]
    split_file: str
    split_type: str
    default_loss: str
    use_class_weights: bool
    recommended_epochs: int
    group_column: Optional[str] = None
    k_folds: int = 5


@dataclass
class ExperimentConfig:
    """Full training and experiment configuration."""

    modality: str
    task: str
    num_classes: int
    class_names: List[str]
    split_file: str
    split_type: str = "fixed"  # "fixed" or "kfold"
    k_folds: int = 5
    fold: Optional[int] = None  # None for fixed split, int 0..k-1 for specific fold

    # Model architecture
    model_name: str = "resnet18"  # "resnet18" or "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = False
    in_channels: int = 3
    image_size: Tuple[int, int] = (224, 224)

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    epochs: int = 10
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_macro_f1"  # "val_loss" or "val_macro_f1"

    # Loss & Imbalance
    loss_type: str = "cross_entropy"  # "cross_entropy", "weighted_ce", "bce_with_logits"
    use_class_weights: bool = False
    pos_weight: Optional[List[float]] = None

    # Reproducibility & Device
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    dataset_version: str = "2.0.0"

    # Paths
    output_dir: str = field(default_factory=lambda: str(OUTPUTS_DIR / "experiments"))
    checkpoint_dir: str = field(default_factory=lambda: str(MODELS_DIR))

    # Notes
    notes: str = "V2.0 baseline model experiment"

    @classmethod
    def from_modality(
        cls,
        modality: str,
        model_name: str = "resnet18",
        epochs: Optional[int] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        fold: Optional[int] = None,
        device: str = "cpu",
        **overrides,
    ) -> ExperimentConfig:
        if modality not in MODALITY_METADATA:
            raise ValueError(f"Unknown modality '{modality}'. Supported: {list(MODALITY_METADATA.keys())}")

        meta = MODALITY_METADATA[modality]
        num_epochs = epochs if epochs is not None else meta["recommended_epochs"]

        config = cls(
            modality=modality,
            task=meta["task"],
            num_classes=meta["num_classes"],
            class_names=meta["class_names"],
            split_file=meta["split_file"],
            split_type=meta["split_type"],
            k_folds=meta.get("k_folds", 5),
            fold=fold,
            model_name=model_name,
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=meta["default_loss"],
            use_class_weights=meta["use_class_weights"],
            device=device,
        )

        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

        return config

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        return d

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
