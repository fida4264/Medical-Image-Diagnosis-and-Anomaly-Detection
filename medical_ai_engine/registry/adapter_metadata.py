"""
Medical AI Engine — Adapter Metadata Dataclass

Tracks metadata for registered disease prediction adapters:
modality, supported classes, model version, framework, creation date, status, health, model path, export formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AdapterMetadata:
    """
    Metadata representation for a disease model adapter.
    """

    modality: str
    supported_classes: List[str] = field(default_factory=list)
    model_version: str = "0.14.2.2"
    framework: str = "PyTorch"
    creation_date: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    status: str = "REGISTERED"  # REGISTERED | LOADED | ERRORED | UNLOADED
    health: str = "HEALTHY"     # HEALTHY | UNHEALTHY | UNKNOWN
    model_path: Optional[str] = None
    label_map_path: Optional[str] = None
    export_formats: List[str] = field(default_factory=lambda: ["PyTorch", "ONNX", "TorchScript"])
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "modality": self.modality,
            "supported_classes": list(self.supported_classes),
            "num_classes": len(self.supported_classes),
            "model_version": self.model_version,
            "framework": self.framework,
            "creation_date": self.creation_date,
            "status": self.status,
            "health": self.health,
            "model_path": self.model_path,
            "label_map_path": self.label_map_path,
            "export_formats": list(self.export_formats),
            "extra": self.extra,
        }
