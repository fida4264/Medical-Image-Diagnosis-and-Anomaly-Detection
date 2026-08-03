"""
Medical AI Engine — Explainability Context

Module 15.1: Explainability Framework Architecture

Shared context object containing all state and inputs required during
an explainability generation lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image


@dataclass
class ExplainabilityContext:
    """
    Shared mutable execution context passed through the ExplainabilityPipeline.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str
    session_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ── Input Data ────────────────────────────────────────────────────────────
    image: Optional[Image.Image] = None
    input_tensor: Optional[Any] = None
    modality: str = "unknown"
    predicted_class: str = "unknown"
    predicted_class_index: int = 0
    confidence: float = 0.0

    # ── Model & Layer References ──────────────────────────────────────────────
    model_reference: Optional[Any] = None
    target_layer_name: Optional[str] = None
    target_layer: Optional[Any] = None
    cam_method_key: str = "gradcam"

    # ── Intermediate Pipeline Outputs ─────────────────────────────────────────
    activation_maps: Optional[np.ndarray] = None
    heatmap_array: Optional[np.ndarray] = None
    overlay_image: Optional[Image.Image] = None

    # ── Saved File Paths ──────────────────────────────────────────────────────
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    activation_map_path: Optional[str] = None

    # ── Clinical & Region Metadata ───────────────────────────────────────────
    important_regions: List[Dict[str, Any]] = field(default_factory=list)
    clinical_notes: List[str] = field(default_factory=list)

    # ── Execution Tracking ────────────────────────────────────────────────────
    execution_state: str = "CREATED"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    future_extensions: Dict[str, Any] = field(default_factory=dict)

    def record_metric(self, name: str, value_ms: float) -> None:
        """Record timing metric in milliseconds."""
        self.performance_metrics[name] = round(value_ms, 3)

    def add_warning(self, warning: str) -> None:
        """Record non-fatal warning message."""
        self.warnings.append(warning)

    def add_error(self, stage: str, error: str) -> None:
        """Record error details."""
        self.errors.append({"stage": stage, "error": error})

    def log(self, message: str, stage: str = "ExplainabilityContext") -> None:
        """Log audit message to notes."""
        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        self.clinical_notes.append(f"[{ts}] [{stage}] {message}")
