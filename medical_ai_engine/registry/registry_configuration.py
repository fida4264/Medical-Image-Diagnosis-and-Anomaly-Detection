"""
Medical AI Engine — Registry Configuration

Defines configuration options for the ModelRegistry (checkpoint paths, lazy loading, caching options).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RegistryConfiguration:
    """
    Configuration options for ModelRegistry.
    """

    enable_lazy_loading: bool = True
    enable_adapter_caching: bool = True
    allow_duplicate_registration: bool = False
    auto_discover_adapters: bool = True

    # Per-modality configuration dict (checkpoint path, label map path)
    modality_routes: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "brain_mri": {
                "checkpoint": "saved_models/brain_mri/best_model.pt",
                "label_map": "metadata/brain_mri_label_map.json",
            },
            "chest_xray": {
                "checkpoint": "saved_models/chest_xray/best_model.pt",
                "label_map": "metadata/chest_xray_label_map.json",
            },
            "brain_ct": {
                "checkpoint": "saved_models/brain_ct/best_model.pt",
                "label_map": "metadata/brain_ct_label_map.json",
            },
            "histopathology": {
                "checkpoint": "saved_models/histopathology/best_model.pt",
                "label_map": "metadata/histopathology_label_map.json",
            },
            "retina": {
                "checkpoint": "saved_models/retina/best_model.pt",
                "label_map": "metadata/retina_label_map.json",
            },
            "skin": {
                "checkpoint": "saved_models/skin/best_model.pt",
                "label_map": "metadata/skin_label_map.json",
            },
            "ultrasound": {
                "checkpoint": "saved_models/ultrasound/best_model.pt",
                "label_map": "metadata/ultrasound_label_map.json",
            },
        }
    )

    def get_modality_config(self, modality: str) -> Dict[str, str]:
        """Return checkpoint and label map paths for a given modality."""
        return self.modality_routes.get(modality.lower(), {})
