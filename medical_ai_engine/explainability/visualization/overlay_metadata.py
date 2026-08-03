"""
Medical AI Engine — Overlay Metadata

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Dataclass recording complete visualization parameters, quality metrics, region count,
spatial dimensions, and export paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class OverlayMetadata:
    """
    Metadata descriptor for a generated visualization overlay artifact bundle.
    """

    request_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    colormap: str = "inferno"
    alpha: float = 0.5
    mask_threshold: float = 0.2

    image_width: int = 224
    image_height: int = 224

    region_count: int = 0
    max_activation_peak: float = 0.0

    quality_metrics: Dict[str, float] = field(default_factory=dict)

    heatmap_file: str = ""
    overlay_file: str = ""
    attention_mask_file: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
