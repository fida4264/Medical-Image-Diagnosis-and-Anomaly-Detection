"""
Medical AI Engine — Explainability Framework Package

Module 15.1: Explainability Framework Architecture

Provides the complete plugin-based Explainable AI (XAI) subsystem for the
Medical AI Platform. Supports Grad-CAM, Grad-CAM++, LayerCAM, ScoreCAM,
EigenCAM, Integrated Gradients, and Guided Backpropagation (implementations
provided in Module 15.2+).

Public API:
    ExplainabilityEngine   — Primary entry point
    ExplainabilityContext  — Shared execution context
    ExplainabilityResult   — Immutable output dataclass
    ExplainabilityConfiguration — Configuration parameters
    CAMRegistry            — Plugin registry
    CAMFactory             — Instance factory
    BaseCAMMethod          — Abstract base for all CAM methods
    ExplainabilityPipeline — 6-stage orchestration pipeline
    ExplainabilityEventListener — Event bus subscriber
"""

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration
from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
from medical_ai_engine.explainability.explainability_pipeline import ExplainabilityPipeline
from medical_ai_engine.explainability.explainability_event_listener import ExplainabilityEventListener
from medical_ai_engine.explainability.cam_registry import CAMRegistry
from medical_ai_engine.explainability.cam_factory import CAMFactory
from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod
from medical_ai_engine.explainability.target_layer_manager import TargetLayerManager
from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
from medical_ai_engine.explainability.overlay_generator import OverlayGenerator
from medical_ai_engine.explainability.clinical_explanation_builder import ClinicalExplanationBuilder
from medical_ai_engine.explainability.explainability_json_builder import ExplainabilityJSONBuilder

__all__ = [
    "ExplainabilityEngine",
    "ExplainabilityContext",
    "ExplainabilityResult",
    "ExplainabilityConfiguration",
    "ExplainabilityPipeline",
    "ExplainabilityEventListener",
    "CAMRegistry",
    "CAMFactory",
    "BaseCAMMethod",
    "TargetLayerManager",
    "HeatmapGenerator",
    "OverlayGenerator",
    "ClinicalExplanationBuilder",
    "ExplainabilityJSONBuilder",
]

__version__ = "0.15.1.0"
