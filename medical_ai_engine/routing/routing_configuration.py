"""
Medical AI Engine — Routing Configuration

Parses and validates configs/routing.yaml or accepts programmatic dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from medical_ai_engine.routing.routing_exceptions import RoutingConfigurationException

logger = logging.getLogger("RoutingConfiguration")


@dataclass
class RoutingConfiguration:
    """
    Configuration manager for the Unified Modality Router.
    """

    config_path: str = "configs/routing.yaml"

    # Router Core
    router_name: str = "UnifiedModalityRouter"
    router_version: str = "1.0.0"
    routing_timeout_ms: float = 5000.0
    allow_manual_override: bool = True
    top_k: int = 3
    enable_routing_audit_log: bool = True

    # Per-modality thresholds
    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "brain_mri": 0.90,
            "chest_xray": 0.85,
            "brain_ct": 0.88,
            "histopathology": 0.92,
            "retina": 0.88,
            "skin": 0.85,
            "ultrasound": 0.83,
        }
    )

    # Fallback rules
    global_confidence_floor: float = 0.60
    below_threshold_action: str = "low_confidence_warn"
    unsupported_modality_action: str = "reject"
    adapter_unavailable_action: str = "reject"

    # Supported Modalities
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri",
            "chest_xray",
            "brain_ct",
            "histopathology",
            "retina",
            "skin",
            "ultrasound",
        ]
    )

    # Modality Classifier
    modality_classifier_checkpoint: str = "outputs/models/modality_classifier_best.pt"
    modality_classifier_label_map: str = "metadata/modality_label_map.json"
    modality_classifier_device: Optional[str] = None
    modality_classifier_target_size: tuple[int, int] = (224, 224)

    # Policies
    warn_on_low_confidence: bool = True
    confidence_warn_threshold: float = 0.70

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> RoutingConfiguration:
        """Load configuration from YAML file or return defaults if missing."""
        path_str = config_path or "configs/routing.yaml"
        path = Path(path_str)

        if not path.is_file():
            logger.warning(f"Routing config file '{path_str}' not found. Using default values.")
            return cls(config_path=path_str)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise RoutingConfigurationException(
                f"Failed to parse routing YAML configuration: {e}",
                config_path=path_str,
                cause=e,
            )

        router_cfg = data.get("router", {})
        thresh_cfg = data.get("confidence_thresholds", {})
        fallback_cfg = data.get("fallback", {})
        supp_cfg = data.get("supported_modalities", [])
        mc_cfg = data.get("modality_classifier", {})
        policy_cfg = data.get("routing_policy", {})

        inst = cls(
            config_path=path_str,
            router_name=router_cfg.get("name", "UnifiedModalityRouter"),
            router_version=router_cfg.get("version", "1.0.0"),
            routing_timeout_ms=float(router_cfg.get("routing_timeout_ms", 5000.0)),
            allow_manual_override=bool(router_cfg.get("allow_manual_override", True)),
            top_k=int(router_cfg.get("top_k", 3)),
            enable_routing_audit_log=bool(router_cfg.get("enable_routing_audit_log", True)),
            confidence_thresholds=thresh_cfg or cls().confidence_thresholds,
            global_confidence_floor=float(fallback_cfg.get("global_confidence_floor", 0.60)),
            below_threshold_action=fallback_cfg.get("below_threshold_action", "low_confidence_warn"),
            unsupported_modality_action=fallback_cfg.get("unsupported_modality_action", "reject"),
            adapter_unavailable_action=fallback_cfg.get("adapter_unavailable_action", "reject"),
            supported_modalities=supp_cfg or cls().supported_modalities,
            modality_classifier_checkpoint=mc_cfg.get("checkpoint_path", "outputs/models/modality_classifier_best.pt"),
            modality_classifier_label_map=mc_cfg.get("label_map_path", "metadata/modality_label_map.json"),
            modality_classifier_device=mc_cfg.get("device"),
            modality_classifier_target_size=tuple(mc_cfg.get("target_size", [224, 224])),
            warn_on_low_confidence=bool(policy_cfg.get("warn_on_low_confidence", True)),
            confidence_warn_threshold=float(policy_cfg.get("confidence_warn_threshold", 0.70)),
        )
        return inst

    def get_threshold(self, modality: str) -> float:
        """Get per-modality confidence threshold with fallback to floor."""
        key = modality.lower()
        return self.confidence_thresholds.get(key, self.global_confidence_floor)

    def is_supported(self, modality: str) -> bool:
        """Check if modality is in supported list."""
        return modality.lower() in [m.lower() for m in self.supported_modalities]
