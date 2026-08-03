"""
Medical AI Engine — CAM Plugin Metadata

Module 15.2: CAM Registry & Plugin System

Defines the `CAMPluginMetadata` dataclass that fully describes a registered
CAM method for discovery, compatibility filtering, and health reporting.

Fields track:
  - Identity         (method_name, version, author, status)
  - Algorithm traits (requires_gradients, heatmap_support, overlay_support)
  - Runtime support  (gpu_support, cpu_support, supported_frameworks)
  - Modality scope   (supported_modalities, target_layer_compatibility)
  - Lifecycle        (registration_timestamp, lifecycle_state)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class PluginStatus(str, Enum):
    """Deployment status of a registered CAM plugin."""
    EXPERIMENTAL = "EXPERIMENTAL"
    BETA         = "BETA"
    STABLE       = "STABLE"
    DEPRECATED   = "DEPRECATED"
    DISABLED     = "DISABLED"


class PluginLifecycleState(str, Enum):
    """
    Lifecycle FSM states for a CAM plugin.

    Valid transitions:
        UNREGISTERED → REGISTERED → INITIALIZED → READY → ACTIVE → UNLOADED
                                                                   ↑
                                                            RELOADED (re-enters READY)
    """
    UNREGISTERED = "UNREGISTERED"
    REGISTERED   = "REGISTERED"
    INITIALIZED  = "INITIALIZED"
    READY        = "READY"
    ACTIVE       = "ACTIVE"
    UNLOADED     = "UNLOADED"
    RELOADED     = "RELOADED"

    @classmethod
    def valid_transitions(cls) -> dict[str, list[str]]:
        """Returns the allowed forward transitions for each state."""
        return {
            cls.UNREGISTERED: [cls.UNREGISTERED, cls.REGISTERED],
            cls.REGISTERED:   [cls.REGISTERED, cls.INITIALIZED, cls.UNLOADED],
            cls.INITIALIZED:  [cls.INITIALIZED, cls.READY, cls.UNLOADED],
            cls.READY:        [cls.READY, cls.INITIALIZED, cls.ACTIVE, cls.UNLOADED],
            cls.ACTIVE:       [cls.ACTIVE, cls.READY, cls.UNLOADED],
            cls.UNLOADED:     [cls.UNLOADED, cls.RELOADED, cls.REGISTERED],
            cls.RELOADED:     [cls.RELOADED, cls.READY, cls.INITIALIZED],
        }

    def can_transition_to(self, target: PluginLifecycleState) -> bool:
        allowed = self.valid_transitions().get(self, [])
        return target in allowed


@dataclass
class CAMPluginMetadata:
    """
    Complete metadata record for a registered CAM plugin.

    Designed as the single source of truth for plugin discovery,
    compatibility filtering, health reporting, and future hot-swap support.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    method_name: str
    version: str = "0.0.0"
    author: str = "Medical AI Platform Team"
    description: str = ""
    status: PluginStatus = PluginStatus.EXPERIMENTAL

    # ── Algorithm Traits ──────────────────────────────────────────────────────
    requires_gradients: bool = True
    heatmap_support: bool = True
    overlay_support: bool = True
    is_stub: bool = True          # True until a real algorithm is implemented

    # ── Runtime Support ───────────────────────────────────────────────────────
    gpu_support: bool = True
    cpu_support: bool = True
    supported_frameworks: list[str] = field(default_factory=lambda: ["pytorch"])

    # ── Modality Scope ────────────────────────────────────────────────────────
    supported_modalities: list[str] = field(default_factory=lambda: [
        "brain_mri", "chest_xray", "skin_lesion",
        "eye_retinal", "breast_mri", "ct_scan", "pathology",
    ])

    # ── Layer Compatibility ───────────────────────────────────────────────────
    target_layer_compatibility: list[str] = field(default_factory=lambda: [
        "Conv2d", "Sequential", "Linear",
    ])

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    lifecycle_state: PluginLifecycleState = PluginLifecycleState.UNREGISTERED
    registration_timestamp: float = field(default_factory=time.time)
    last_used_timestamp: float = 0.0
    use_count: int = 0

    # ── Module Path (for lazy loading) ────────────────────────────────────────
    module_path: str = ""       # e.g. "medical_ai_engine.explainability.methods.gradcam"
    class_name: str = ""        # e.g. "GradCAMMethod"

    # ── Extra ─────────────────────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["lifecycle_state"] = self.lifecycle_state.value
        return d

    def is_ready(self) -> bool:
        """True when the plugin is in a state where it can serve requests."""
        return self.lifecycle_state in (
            PluginLifecycleState.READY,
            PluginLifecycleState.ACTIVE,
            PluginLifecycleState.RELOADED,
        )

    def supports_modality(self, modality: str) -> bool:
        return modality in self.supported_modalities

    def supports_framework(self, framework: str) -> bool:
        return framework.lower() in [f.lower() for f in self.supported_frameworks]

    def record_use(self) -> None:
        self.use_count += 1
        self.last_used_timestamp = time.time()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CAMPluginMetadata("
            f"method_name={self.method_name!r}, "
            f"version={self.version!r}, "
            f"status={self.status.value!r}, "
            f"lifecycle_state={self.lifecycle_state.value!r})"
        )


# ── Pre-built metadata descriptors for the 7 built-in plugin types ─────────────
# These describe what will be registered — no algorithm is implemented here.

BUILTIN_PLUGIN_METADATA: dict[str, CAMPluginMetadata] = {
    "gradcam": CAMPluginMetadata(
        method_name="gradcam",
        version="0.15.3.0",
        description="Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017). "
                    "Produces class-discriminative localization maps using gradients of any "
                    "target concept flowing into the final convolutional layer.",
        requires_gradients=True,
        heatmap_support=True,
        overlay_support=True,
        gpu_support=True,
        cpu_support=True,
        is_stub=True,
        status=PluginStatus.STABLE,
        module_path="medical_ai_engine.explainability.methods.gradcam",
        class_name="GradCAMMethod",
        tags=["gradient", "cnn", "cam", "localization"],
    ),
    "gradcam_plus_plus": CAMPluginMetadata(
        method_name="gradcam_plus_plus",
        version="0.15.3.0",
        description="Grad-CAM++ (Chattopadhay et al., 2018). Improved generalization over "
                    "multiple occurrences of same object class in an image.",
        requires_gradients=True,
        is_stub=True,
        status=PluginStatus.STABLE,
        module_path="medical_ai_engine.explainability.methods.gradcam_plus_plus",
        class_name="GradCAMPlusPlusMethod",
        tags=["gradient", "cnn", "cam", "localization", "multi-object"],
    ),
    "layercam": CAMPluginMetadata(
        method_name="layercam",
        version="0.15.3.0",
        description="LayerCAM (Jiang et al., 2021). Multi-layer spatial activation maps "
                    "showing fine-grained discriminative regions at shallow and deep layers.",
        requires_gradients=True,
        is_stub=True,
        status=PluginStatus.BETA,
        module_path="medical_ai_engine.explainability.methods.layercam",
        class_name="LayerCAMMethod",
        tags=["gradient", "cnn", "cam", "multi-layer", "fine-grained"],
    ),
    "scorecam": CAMPluginMetadata(
        method_name="scorecam",
        version="0.15.3.0",
        description="Score-CAM (Wang et al., 2020). Gradient-free activation-based map "
                    "using forward-pass channel scores. No backpropagation required.",
        requires_gradients=False,
        is_stub=True,
        status=PluginStatus.STABLE,
        module_path="medical_ai_engine.explainability.methods.scorecam",
        class_name="ScoreCAMMethod",
        tags=["gradient-free", "cnn", "cam", "forward-pass"],
    ),
    "eigencam": CAMPluginMetadata(
        method_name="eigencam",
        version="0.15.3.0",
        description="EigenCAM (Muhammad & Yeasin, 2020). Uses principal components of "
                    "activation maps. Gradient-free, fast, suitable for real-time use.",
        requires_gradients=False,
        is_stub=True,
        status=PluginStatus.STABLE,
        module_path="medical_ai_engine.explainability.methods.eigencam",
        class_name="EigenCAMMethod",
        tags=["gradient-free", "pca", "cam", "fast", "real-time"],
    ),
    "integrated_gradients": CAMPluginMetadata(
        method_name="integrated_gradients",
        version="0.15.3.0",
        description="Integrated Gradients (Sundararajan et al., 2017). Axiomatic attribution "
                    "method satisfying completeness and sensitivity. Requires baseline input.",
        requires_gradients=True,
        is_stub=True,
        status=PluginStatus.BETA,
        module_path="medical_ai_engine.explainability.methods.integrated_gradients",
        class_name="IntegratedGradientsMethod",
        tags=["gradient", "attribution", "axiomatic", "baseline"],
    ),
    "guided_backprop": CAMPluginMetadata(
        method_name="guided_backprop",
        version="0.15.3.0",
        description="Guided Backpropagation (Springenberg et al., 2015). Backpropagates "
                    "only positive gradients through ReLU activations for sharp visualizations.",
        requires_gradients=True,
        is_stub=True,
        status=PluginStatus.BETA,
        module_path="medical_ai_engine.explainability.methods.guided_backprop",
        class_name="GuidedBackpropMethod",
        tags=["gradient", "deconvolution", "sharp", "visualization"],
    ),
}
