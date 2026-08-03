"""
Medical AI Engine — Explainability Configuration

Module 15.1: Explainability Framework Architecture

Central configuration dataclass for the Explainability Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExplainabilityConfiguration:
    """
    Configuration parameters for the Explainability Engine and CAM algorithms.
    """

    # ── Engine Core Settings ──────────────────────────────────────────────────
    enabled: bool = True
    default_cam_method: str = "gradcam"
    output_directory: str = "outputs/explainability"
    save_heatmaps: bool = True
    save_overlays: bool = True
    save_activation_maps: bool = False

    # ── Visualization Styling ─────────────────────────────────────────────────
    colormap: str = "JET"                  # OpenCV colormap (JET, VIRIDIS, INFERNO, HOT)
    overlay_alpha: float = 0.5            # Heatmap transparency blending weight (0.0 to 1.0)
    image_size: tuple[int, int] = (224, 224)
    interpolate_mode: str = "bilinear"

    # ── Target Layer Resolution Rules per Modality ──────────────────────────────
    default_target_layers: Dict[str, str] = field(default_factory=lambda: {
        "brain_mri": "layer4",
        "chest_xray": "features",
        "brain_ct": "layer4",
        "histopathology": "layer4",
        "retina": "features",
        "skin": "features",
        "ultrasound": "layer4",
    })

    # ── Quality & Thresholds ──────────────────────────────────────────────────
    min_confidence_for_explanation: float = 0.30
    auto_generate_on_prediction: bool = True
    max_important_regions: int = 3
    region_threshold_pct: float = 0.50     # Pixels > 50% max intensity counted as important region

    # ── Supported Methods Registry Keys ───────────────────────────────────────
    supported_methods: List[str] = field(default_factory=lambda: [
        "gradcam",
        "gradcam_plus_plus",
        "layercam",
        "scorecam",
        "eigencam",
        "integrated_gradients",
        "guided_backprop",
    ])

    @classmethod
    def default(cls) -> "ExplainabilityConfiguration":
        """Return default configuration instance."""
        return cls()

    def get_target_layer_for_modality(self, modality: str) -> str:
        """Resolve target layer name for a specific modality."""
        return self.default_target_layers.get(modality.lower(), "layer4")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "default_cam_method": self.default_cam_method,
            "output_directory": self.output_directory,
            "colormap": self.colormap,
            "overlay_alpha": self.overlay_alpha,
            "min_confidence_for_explanation": self.min_confidence_for_explanation,
            "supported_methods": self.supported_methods,
        }
