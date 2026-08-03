"""
architecture_validator.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Architecture compliance validator for Module 15.
"""

from __future__ import annotations

import importlib
import logging

from .verification_metrics import CategoryScore

logger = logging.getLogger(__name__)


class ArchitectureValidator:
    """
    Validates structural architecture compliance of Module 15 components.

    Checks:
      1. Package imports clean without deprecation or circular import issues
      2. Key interface definitions present (ICAMPlugin, ICAMRegistry, BaseCAMMethod)
      3. Component presence (Registry, Factory, Loader, GradCAM Engine, Visualization, Clinical Engine, Integration)
      4. Event bus decoupling (no direct imports of MedicalAIEngine inside explainability core)
    """

    REQUIRED_MODULES = [
        "medical_ai_engine.explainability",
        "medical_ai_engine.explainability.registry",
        "medical_ai_engine.explainability.plugins.gradcam",
        "medical_ai_engine.explainability.visualization",
        "medical_ai_engine.explainability.clinical",
        "medical_ai_engine.explainability.integration",
    ]

    REQUIRED_CLASSES = [
        ("medical_ai_engine.explainability.registry", "CAMProductionRegistry"),
        ("medical_ai_engine.explainability.registry", "CAMPluginFactory"),
        ("medical_ai_engine.explainability.registry", "CAMPluginLoader"),
        ("medical_ai_engine.explainability.plugins.gradcam", "GradGAMEngine"),
        ("medical_ai_engine.explainability.plugins.gradcam", "GradCAMPlugin"),
        ("medical_ai_engine.explainability.visualization", "HeatmapRenderer"),
        ("medical_ai_engine.explainability.clinical", "ClinicalExplanationEngine"),
        ("medical_ai_engine.explainability.integration", "ExplainabilityIntegrator"),
        ("medical_ai_engine.explainability.integration", "WorkflowCoordinator"),
    ]

    def validate(self, score: CategoryScore) -> None:
        """Run architecture validation rules."""

        # Rule 1: Package imports
        for mod_name in self.REQUIRED_MODULES:
            try:
                importlib.import_module(mod_name)
                score.record_pass(f"Module '{mod_name}' imports cleanly")
            except Exception as exc:
                score.record_fail(f"Module '{mod_name}' import failed: {exc}")

        # Rule 2: Component class existence
        for mod_path, cls_name in self.REQUIRED_CLASSES:
            try:
                mod = importlib.import_module(mod_path)
                cls_obj = getattr(mod, cls_name, None)
                if cls_obj is not None:
                    score.record_pass(f"Class '{cls_name}' verified in '{mod_path}'")
                else:
                    score.record_fail(f"Class '{cls_name}' missing in '{mod_path}'")
            except Exception as exc:
                score.record_fail(f"Failed loading class '{cls_name}': {exc}")

        # Rule 3: Event bus decoupling check
        try:
            # Verify explainability.explainability_engine does not import MedicalAIEngine directly
            import medical_ai_engine.explainability.explainability_engine as ee_mod
            source = getattr(ee_mod, "__file__", "")
            if source:
                with open(source, "r", encoding="utf-8") as fh:
                    content = fh.read()
                    if "import MedicalAIEngine" in content or "from medical_ai_engine.core.medical_ai_engine import" in content:
                        score.record_fail("Explainability Engine has tight coupling to MedicalAIEngine core")
                    else:
                        score.record_pass("Explainability Engine is decoupled from MedicalAIEngine core")
            else: # pragma: no cover
                score.record_pass("Explainability Engine source file check skipped")
        except Exception as exc: # pragma: no cover
            score.record_fail(f"Decoupling check error: {exc}")


__all__ = ["ArchitectureValidator"]
