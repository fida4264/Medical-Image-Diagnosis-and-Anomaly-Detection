"""
report_generator.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Main facade for generating professional medical AI reports.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from medical_ai_engine.reporting.recommendation_builder import RecommendationBuilder
from medical_ai_engine.reporting.report_configuration import ReportConfiguration
from medical_ai_engine.reporting.report_exporter import ReportExporter
from medical_ai_engine.reporting.report_templates import (
    MedicalReportData,
    PatientInfo,
    ProcessingStats,
)
from medical_ai_engine.reporting.reporting_exceptions import ReportGenerationError
from medical_ai_engine.reporting.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Main entry point for generating and exporting Medical AI Reports.

    Fuses outputs from:
    1. PredictionResult (Disease, Confidence, Modality, Request ID)
    2. ExplainabilityResult (GradCAM heatmaps, saliency coverage)
    3. AnomalyResult (OOD score, Unknown Disease detection, Risk level)

    Without modifying any prediction or upstream engine pipelines.
    """

    def __init__(
        self,
        config: Optional[ReportConfiguration] = None,
    ) -> None:
        self.config = config or ReportConfiguration.default()
        self._summary_builder = SummaryBuilder()
        self._recommendation_builder = RecommendationBuilder()
        self._exporter = ReportExporter()

    def generate_report(
        self,
        prediction_result: Any,
        patient_id: Optional[str] = None,
        case_id: Optional[str] = None,
        explainability_result: Optional[Any] = None,
        anomaly_result: Optional[Any] = None,
    ) -> MedicalReportData:
        """
        Generate structured MedicalReportData from engine outputs.

        Parameters
        ----------
        prediction_result : Any
            PredictionResult object or dictionary.
        patient_id : str, optional
            Patient identifier (defaults to "PT-DEMO-001" or prediction.patient_id).
        case_id : str, optional
            Case/study identifier.
        explainability_result : Any, optional
            ExplainabilityResult object or dictionary.
        anomaly_result : Any, optional
            AnomalyResult object or dictionary.

        Returns
        -------
        MedicalReportData
        """
        if prediction_result is None:
            raise ReportGenerationError("prediction_result cannot be None.")

        try:
            # 1. Extract Prediction Details
            disease, confidence, modality, req_id = self._extract_pred(prediction_result)

            # 2. Patient Info
            p_id = patient_id or self._extract_field(prediction_result, "patient_id", "PT-DEMO-001")
            c_id = case_id or f"CASE-{req_id[:8]}"

            patient_info = PatientInfo(
                patient_id=p_id,
                case_id=c_id,
                modality=modality,
            )

            # 3. Build Clinical Summary
            summary = self._summary_builder.build_summary(
                prediction_result=prediction_result,
                explainability_result=explainability_result,
                anomaly_result=anomaly_result,
            )

            # 4. Build Clinical Recommendations
            recommendation = self._recommendation_builder.build_recommendation(
                prediction_result=prediction_result,
                anomaly_result=anomaly_result,
            )

            # 5. Heatmap References
            heatmap_refs = self._extract_heatmap_refs(explainability_result)

            # 6. Processing Statistics
            processing_stats = self._extract_processing_stats(prediction_result, explainability_result, anomaly_result)

            report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"

            logger.info(
                "[ReportGenerator] Generated report %s for patient %s (Disease: %s, Conf: %.1f%%)",
                report_id,
                p_id,
                disease,
                confidence * 100.0 if confidence <= 1.0 else confidence,
            )

            return MedicalReportData(
                report_id=report_id,
                patient_info=patient_info,
                predicted_disease=disease,
                confidence=confidence,
                summary=summary,
                recommendation=recommendation,
                heatmap_references=heatmap_refs if self.config.include_heatmap_references else [],
                processing_stats=processing_stats,
                disclaimer=self.config.disclaimer_text if self.config.include_disclaimer else "",
            )
        except Exception as exc:
            if isinstance(exc, ReportGenerationError):
                raise
            raise ReportGenerationError(f"Failed to generate medical report: {exc}") from exc

    def generate_and_export(
        self,
        prediction_result: Any,
        patient_id: Optional[str] = None,
        case_id: Optional[str] = None,
        explainability_result: Optional[Any] = None,
        anomaly_result: Optional[Any] = None,
        output_directory: Optional[Union[str, Path]] = None,
        file_prefix: str = "sample_medical_report",
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Generate report and export to JSON, Markdown, and PDF.

        Returns
        -------
        Dict[str, Path]
            Mapping format ('json', 'markdown', 'pdf') to output file paths.
        """
        report_data = self.generate_report(
            prediction_result=prediction_result,
            patient_id=patient_id,
            case_id=case_id,
            explainability_result=explainability_result,
            anomaly_result=anomaly_result,
        )

        out_dir = output_directory or self.config.default_output_directory
        export_fmts = formats or self.config.export_formats

        return self._exporter.export(
            report=report_data,
            output_directory=out_dir,
            file_prefix=file_prefix,
            formats=export_fmts,
        )

    # ------------------------------------------------------------------
    # Helper Extraction Methods
    # ------------------------------------------------------------------

    def _extract_pred(self, pred: Any) -> tuple:
        if isinstance(pred, dict):
            disease = pred.get("predicted_disease") or pred.get("predicted_class") or "Unknown Disease"
            conf = float(pred.get("confidence", 0.0))
            mod = pred.get("modality", "chest_xray")
            req = pred.get("request_id", "REQ-DEMO-001")
        else:
            disease = getattr(pred, "predicted_disease", None) or getattr(pred, "predicted_class", "Unknown Disease")
            conf = float(getattr(pred, "confidence", 0.0))
            mod = getattr(pred, "modality", "chest_xray")
            req = getattr(pred, "request_id", "REQ-DEMO-001")
        return disease, conf, mod, req

    def _extract_field(self, obj: Any, field_name: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(field_name, default)
        return getattr(obj, field_name, default)

    def _extract_heatmap_refs(self, exp: Optional[Any]) -> List[str]:
        if exp is None:
            return []
        refs = []
        if isinstance(exp, dict):
            if "heatmap_path" in exp:
                refs.append(str(exp["heatmap_path"]))
            if "overlay_path" in exp:
                refs.append(str(exp["overlay_path"]))
        else:
            if hasattr(exp, "heatmap_path") and exp.heatmap_path:
                refs.append(str(exp.heatmap_path))
            if hasattr(exp, "overlay_path") and exp.overlay_path:
                refs.append(str(exp.overlay_path))

        if not refs:
            refs = ["artifacts/gradcam_overlay_chest_xray.png"]
        return refs

    def _extract_processing_stats(
        self, pred: Any, exp: Optional[Any], anomaly: Optional[Any]
    ) -> ProcessingStats:
        lat_pred = float(self._extract_field(pred, "processing_time_ms", 12.5))
        lat_exp = float(self._extract_field(exp, "processing_time_ms", 8.2)) if exp else 0.0
        lat_anom = float(self._extract_field(anomaly, "processing_time_ms", 4.1)) if anomaly else 0.0
        arch = str(self._extract_field(pred, "model_architecture", "DenseNet-121"))

        total = lat_pred + lat_exp + lat_anom

        return ProcessingStats(
            inference_latency_ms=lat_pred,
            explainability_latency_ms=lat_exp,
            anomaly_latency_ms=lat_anom,
            total_processing_time_ms=total,
            model_architecture=arch,
            engine_version="0.18.0.0",
        )


__all__ = ["ReportGenerator"]
