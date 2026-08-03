"""
Medical AI Engine — ClinicalReportHookStage (Stage 7)

Generates disease-specific clinical diagnostic report dictionary and hooks into
downstream reporting framework. Populates ctx.clinical_report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class ClinicalReportHookStage(AbstractPipelineStage):
    """
    Stage 7: Clinical Report Generation Hook.
    """

    def __init__(self) -> None:
        super().__init__("ClinicalReportHookStage")

    def validate(self, ctx: EngineContext) -> None:
        if ctx.disease_prediction is None:
            raise StageValidationException(
                "ctx.disease_prediction is missing for clinical report generation.",
                stage_name=self.name,
            )
        if ctx.confidence is None:
            raise StageValidationException(
                "ctx.confidence is missing for clinical report generation.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Generating clinical report structure...", stage=self.name)

        modality = (ctx.modality_prediction or {}).get("detected_modality", "unknown")
        disease_pred = ctx.disease_prediction or {}
        dx = disease_pred.get("predicted_class", "unknown")
        conf_data = ctx.confidence or {}

        ctx.clinical_report = {
            "report_type": f"{modality.replace('_', ' ').title()} Diagnostic Report",
            "primary_diagnosis": dx,
            "reliability": conf_data.get("reliability", "unknown"),
            "requires_manual_review": conf_data.get("requires_manual_review", False),
            "differential_diagnoses": disease_pred.get("top_k_predictions", []),
            "clinical_note": (
                f"Automated AI diagnostic analysis for modality '{modality}' "
                f"identified primary diagnosis '{dx}' with confidence {conf_data.get('score', 0.0):.4f} "
                f"({conf_data.get('reliability')} reliability)."
            ),
        }

        ctx.set_timestamp("report_complete")
        ctx.log(
            f"Clinical report generated: primary_diagnosis='{dx}', type='{ctx.clinical_report['report_type']}'",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back ClinicalReportHookStage...", stage=self.name)
        ctx.clinical_report = None
