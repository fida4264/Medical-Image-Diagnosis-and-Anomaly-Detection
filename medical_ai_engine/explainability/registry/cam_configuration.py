"""
Medical AI Engine — CAM Plugin Configuration

Module 15.2: CAM Registry & Plugin System

Defines `CAMPluginConfiguration` — the per-plugin runtime configuration injected
by the factory when instantiating a CAM plugin. Separate from global
`ExplainabilityConfiguration` (Module 15.1) to preserve backward compatibility.

Responsibilities:
  - Hold plugin-specific parameters (batch size, baseline steps, etc.)
  - Carry target layer override
  - Hold output resolution parameters
  - Support dict → config round-trip for serialization
  - Provide typed defaults for every supported method
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CAMPluginConfiguration:
    """
    Per-plugin runtime configuration injected by CAMFactory.

    Plugin implementations MUST NOT hard-code behaviour that belongs here.
    All method-specific knobs are accessed through this configuration object.
    """

    # ── Target Layer ──────────────────────────────────────────────────────────
    target_layer_name: str = "layer4"
    """Name of the model layer to hook. Resolved by TargetLayerManager if empty."""

    # ── Output Resolution ─────────────────────────────────────────────────────
    output_height: int = 224
    output_width: int = 224

    # ── Grad-CAM / Grad-CAM++ / LayerCAM specific ────────────────────────────
    target_category: int | None = None
    """Prediction class index to explain. None → use the argmax of the output."""

    smooth_gradients: bool = False
    smooth_samples: int = 25
    smooth_noise: float = 0.05

    # ── Score-CAM specific ────────────────────────────────────────────────────
    batch_size: int = 32
    """Batch size for ScoreCAM forward passes."""

    # ── Integrated Gradients specific ─────────────────────────────────────────
    n_steps: int = 50
    """Number of integration steps (Riemann approximation)."""

    baseline_type: str = "zero"
    """Baseline input: 'zero', 'noise', 'blur'."""

    # ── Post-processing ───────────────────────────────────────────────────────
    relu_on_output: bool = True
    """Apply ReLU to final activation map (keeps only positive contributions)."""

    normalize_output: bool = True
    """Normalize the activation map to [0, 1]."""

    # ── Framework ─────────────────────────────────────────────────────────────
    device: str = "cpu"
    """Torch device string: 'cpu' | 'cuda' | 'cuda:0'."""

    # ── Extra pass-through parameters ────────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CAMPluginConfiguration:
        """Construct from a dictionary, ignoring unknown keys safely."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def default_for_method(cls, method_key: str) -> CAMPluginConfiguration:
        """
        Return a configuration instance with sensible defaults for the given method.

        This factory method centralises all method-specific default reasoning
        so plugin implementations never hard-code their own defaults.
        """
        defaults: dict[str, dict[str, Any]] = {
            "gradcam": {
                "smooth_gradients": False,
                "relu_on_output": True,
            },
            "gradcam_plus_plus": {
                "smooth_gradients": True,
                "smooth_samples": 20,
                "relu_on_output": True,
            },
            "layercam": {
                "relu_on_output": True,
            },
            "scorecam": {
                "batch_size": 32,
                "relu_on_output": True,
            },
            "eigencam": {
                "relu_on_output": False,  # EigenCAM already non-negative
            },
            "integrated_gradients": {
                "n_steps": 50,
                "baseline_type": "zero",
                "relu_on_output": False,
            },
            "guided_backprop": {
                "relu_on_output": False,
            },
        }
        method_defaults = defaults.get(method_key.lower(), {})
        return cls(**method_defaults)

    def with_device(self, device: str) -> CAMPluginConfiguration:
        """Return a copy of this config with the device overridden."""
        import dataclasses
        return dataclasses.replace(self, device=device)

    def with_target_layer(self, layer_name: str) -> CAMPluginConfiguration:
        """Return a copy of this config with a specific target layer."""
        import dataclasses
        return dataclasses.replace(self, target_layer_name=layer_name)
