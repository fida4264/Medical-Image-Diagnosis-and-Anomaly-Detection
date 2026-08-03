"""
integration_validator.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — System integration validator for end-to-end XAI workflow.
"""

from __future__ import annotations

import logging
from PIL import Image

from medical_ai_engine.explainability.integration import ExplainabilityIntegrator
from .verification_metrics import CategoryScore

logger = logging.getLogger(__name__)


class SystemIntegrationValidator:
    """
    Validates end-to-end integration between MedicalAIEngine and Explainability Framework.
    """

    def validate(self, score: CategoryScore) -> None:
        """Run integration verification checks across modalities."""

        try:
            integrator = ExplainabilityIntegrator()

            # Test 1: Health check
            health = integrator.health_check()
            if health.get("enabled", False):
                score.record_pass("ExplainabilityIntegrator is active and healthy")
            else:
                score.record_fail("ExplainabilityIntegrator health check returned disabled")

            # Test 2: Execution across sample modalities
            dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
            test_modalities = ["brain_mri", "chest_xray", "skin"]

            for mod in test_modalities:
                res = integrator.execute_with_explanation(
                    image_source=dummy_img,
                    manual_modality_override=mod,
                )
                if res.is_success and res.gradcam_result is not None:
                    score.record_pass(f"Integration pipeline executed cleanly for '{mod}'")
                else:
                    score.record_fail(f"Integration pipeline failed for '{mod}'")

        except Exception as exc:
            score.record_fail(f"Integration validation exception: {exc}")


__all__ = ["SystemIntegrationValidator"]
