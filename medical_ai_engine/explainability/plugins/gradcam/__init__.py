"""
Medical AI Engine — Grad-CAM Engine Package

Module 15.3: Grad-CAM Engine

Exports:
    GradCAMPlugin         — Production ICAMPlugin implementation
    GradGAMEngine         — Execution pipeline orchestrator
    GradCAMConfiguration  — Parameter dataclass
    ActivationExtractor   — Forward hook capture
    GradientHandler       — Backward hook capture
    TargetSelector        — Classification target category resolver
    ModelHookManager      — Hook lifecycle context manager
    CAMComputation        — Mathematical weighting & linear combination
    HeatmapNormalizer     — Resizing & normalization
"""

from medical_ai_engine.explainability.plugins.gradcam.gradcam_exceptions import (
    GradCAMException,
    MissingTargetLayerError,
    UnsupportedModelError,
    HookRegistrationError,
    GradientExtractionError,
    InvalidTensorShapeError,
    CAMComputationError,
)
from medical_ai_engine.explainability.plugins.gradcam.gradcam_configuration import (
    GradCAMConfiguration,
)
from medical_ai_engine.explainability.plugins.gradcam.model_hook_manager import (
    ModelHookManager,
)
from medical_ai_engine.explainability.plugins.gradcam.activation_extractor import (
    ActivationExtractor,
)
from medical_ai_engine.explainability.plugins.gradcam.gradient_handler import (
    GradientHandler,
)
from medical_ai_engine.explainability.plugins.gradcam.target_selector import (
    TargetSelector,
)
from medical_ai_engine.explainability.plugins.gradcam.cam_computation import (
    CAMComputation,
)
from medical_ai_engine.explainability.plugins.gradcam.heatmap_normalizer import (
    HeatmapNormalizer,
)
from medical_ai_engine.explainability.plugins.gradcam.gradcam_engine import (
    GradGAMEngine,
)
from medical_ai_engine.explainability.plugins.gradcam.gradcam_plugin import (
    GradCAMPlugin,
)


def _auto_register() -> None:
    """Auto-register production GradCAMPlugin into CAMProductionRegistry."""
    try:
        from medical_ai_engine.explainability.registry.cam_registry import CAMProductionRegistry
        reg = CAMProductionRegistry.get_instance()
        reg.register("gradcam", GradCAMPlugin, overwrite=True)
    except Exception:
        pass


_auto_register()

__all__ = [
    "GradCAMPlugin",
    "GradGAMEngine",
    "GradCAMConfiguration",
    "ActivationExtractor",
    "GradientHandler",
    "TargetSelector",
    "ModelHookManager",
    "CAMComputation",
    "HeatmapNormalizer",
    "GradCAMException",
    "MissingTargetLayerError",
    "UnsupportedModelError",
    "HookRegistrationError",
    "GradientExtractionError",
    "InvalidTensorShapeError",
    "CAMComputationError",
]
