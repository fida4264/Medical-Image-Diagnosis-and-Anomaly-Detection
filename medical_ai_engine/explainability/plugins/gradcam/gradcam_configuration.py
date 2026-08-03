"""
Medical AI Engine — Grad-CAM Configuration

Module 15.3: Grad-CAM Engine

Per-instance configuration parameters specifically for the Grad-CAM engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GradCAMConfiguration:
    """
    Configuration parameters for Grad-CAM execution.
    """

    target_layer_name: str = ""
    """Layer name override. If empty, resolved via TargetLayerManager."""

    target_category: int | None = None
    """Class index to explain. If None, uses argmax of logits/output."""

    relu_on_output: bool = True
    """Apply ReLU activation to heatmaps (keeps positive contributions)."""

    normalize_output: bool = True
    """Normalize final activation map to [0, 1]."""

    output_height: int = 224
    """Output heatmap spatial height."""

    output_width: int = 224
    """Output heatmap spatial width."""

    detach_tensors: bool = True
    """Detach tensors from PyTorch computational graph after capture."""

    retain_graph: bool = False
    """Retain computational graph after backward pass."""

    device: str = "cpu"
    """Target execution device ('cpu', 'cuda', 'cuda:0')."""

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def default(cls) -> GradCAMConfiguration:
        return cls()
