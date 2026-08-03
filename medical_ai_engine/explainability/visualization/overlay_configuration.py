"""
Medical AI Engine — Overlay Configuration

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Configuration parameters governing colormaps, alpha blending ratios, attention mask
thresholding, bounding box rendering, and image artifact export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Tuple


SUPPORTED_COLORMAPS = ("inferno", "jet", "turbo", "viridis", "magma")


@dataclass
class OverlayConfiguration:
    """
    Configuration parameters for heatmap rendering and clinical overlay generation.
    """

    colormap: str = "inferno"
    """Colormap name: 'inferno' (default), 'jet', 'turbo', 'viridis', 'magma'."""

    alpha: float = 0.5
    """Alpha transparency ratio for blending [0.1, 0.9]. Default: 0.5."""

    mask_threshold: float = 0.2
    """Intensity threshold [0.0, 1.0] for generating attention mask and region extraction."""

    draw_bounding_boxes: bool = True
    """Draw bounding boxes around high-activation connected components."""

    bounding_box_color: Tuple[int, int, int] = (255, 0, 0)
    """RGB color tuple for bounding boxes (default: Red)."""

    bounding_box_thickness: int = 2
    """Line thickness for bounding boxes in pixels."""

    export_format: str = "PNG"
    """Image export format: 'PNG' or 'JPEG'."""

    export_dir: str = ""
    """Export directory path for output image artifacts."""

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cmap = self.colormap.lower()
        if cmap not in SUPPORTED_COLORMAPS:
            raise ValueError(
                f"Unsupported colormap '{self.colormap}'. Must be one of {SUPPORTED_COLORMAPS}."
            )
        self.colormap = cmap

        if not (0.05 <= self.alpha <= 0.95):
            raise ValueError(f"Alpha transparency must be between 0.1 and 0.9. Got {self.alpha}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def default(cls) -> OverlayConfiguration:
        return cls()
