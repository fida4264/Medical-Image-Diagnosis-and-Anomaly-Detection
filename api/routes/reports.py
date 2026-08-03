"""
api/routes/reports.py — Module 21: Production REST API
Medical report metadata and file download endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse

from api.dependencies import (
    get_query_engine,
    get_report_exporter,
    verify_api_key,
)
from api.exceptions import PredictionNotFoundException, ReportExportException
from api.schemas import ReportMetadataResponse
from medical_ai_engine.database.query_engine import QueryEngine
from medical_ai_engine.reporting.report_exporter import ReportExporter

router = APIRouter(prefix="", tags=["Medical Reports"])


def _build_report_from_record(rec):
    """Build a MedicalReportData object directly from a PredictionRecord."""
    from medical_ai_engine.reporting.report_templates import (
        MedicalReportData,
        PatientInfo,
        ProcessingStats,
    )
    from medical_ai_engine.reporting.summary_builder import ClinicalSummary
    from medical_ai_engine.reporting.recommendation_builder import ClinicalRecommendation
    import uuid

    reliability = (
        "HIGH" if rec.confidence >= 0.85
        else ("MODERATE" if rec.confidence >= 0.65 else "LOW")
    )
    is_anomaly = rec.anomaly_status == "UNKNOWN_ANOMALY"
    risk_level = rec.risk_level or ("CRITICAL" if is_anomaly else "LOW")
    conf_pct = round(rec.confidence * 100, 1)

    patient_info = PatientInfo(
        patient_id=rec.patient_id,
        case_id=rec.case_id,
        modality=rec.modality,
    )
    summary = ClinicalSummary(
        primary_finding=rec.predicted_disease,
        diagnostic_impression=f"{rec.predicted_disease} detected with {conf_pct}% confidence ({reliability} reliability).",
        explainability_narrative=rec.explainability_summary or "GradCAM saliency map generated.",
        anomaly_narrative=f"Anomaly status: {rec.anomaly_status}. Risk level: {risk_level}.",
        key_observations=[
            f"Predicted class: {rec.predicted_disease}",
            f"Anomaly classification: {rec.anomaly_status}",
            f"Clinical risk: {risk_level}",
        ],
    )
    recommendation = ClinicalRecommendation(
        action_code=rec.recommended_action,
        recommended_action=rec.recommended_action.replace("_", " ").title(),
        urgency_level="URGENT" if risk_level in ("HIGH", "CRITICAL") else "ROUTINE",
        rationale=f"Automated assessment based on {conf_pct}% confidence {rec.predicted_disease} prediction.",
        suggested_follow_up=f"Clinical follow-up for {rec.predicted_disease} findings recommended.",
    )
    stats = ProcessingStats(
        total_processing_time_ms=rec.processing_time_ms,
    )
    report = MedicalReportData(
        report_id=str(uuid.uuid4()),
        patient_info=patient_info,
        predicted_disease=rec.predicted_disease,
        confidence=rec.confidence,
        summary=summary,
        recommendation=recommendation,
        heatmap_references=[rec.heatmap_path] if rec.heatmap_path else [],
        processing_stats=stats,
    )
    return report


@router.get(
    "/reports/{prediction_id}",
    response_model=ReportMetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Report Metadata for a Prediction",
    description=(
        "Retrieve medical AI diagnostic report metadata, summary, and available "
        "download URLs for a prediction ID."
    ),
)
async def get_report_metadata(
    prediction_id: str,
    query_engine: QueryEngine = Depends(get_query_engine),
    _auth: Optional[str] = Depends(verify_api_key),
) -> ReportMetadataResponse:
    rec = query_engine.get_by_id(prediction_id)
    if not rec:
        raise PredictionNotFoundException(prediction_id)

    reliability = (
        "HIGH" if rec.confidence >= 0.85
        else ("MODERATE" if rec.confidence >= 0.65 else "LOW")
    )
    is_anomaly = rec.anomaly_status == "UNKNOWN_ANOMALY"
    requires_review = rec.recommended_action in ("CLINICAL_REVIEW", "URGENT_REVIEW")

    download_urls = {
        "json": f"/reports/{prediction_id}/download?format=json",
        "md":   f"/reports/{prediction_id}/download?format=md",
        "pdf":  f"/reports/{prediction_id}/download?format=pdf",
    }

    return ReportMetadataResponse(
        prediction_id=prediction_id,
        patient_id=rec.patient_id,
        generated_at=rec.created_at,
        modality=rec.modality,
        primary_diagnosis=rec.predicted_disease,
        confidence=f"{rec.confidence * 100:.1f}%",
        reliability=reliability,
        clinical_note=rec.explainability_summary or f"Automated report for case {prediction_id[:8]}.",
        differential_diagnoses=[],
        requires_manual_review=requires_review,
        anomaly_score=float(rec.risk_score),
        is_anomaly=is_anomaly,
        processing_time_ms=float(rec.processing_time_ms),
        download_urls=download_urls,
    )


@router.get(
    "/reports/{prediction_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Download Generated Medical Report Document",
    description=(
        "Export and download the generated medical report document "
        "in PDF, JSON, or Markdown format."
    ),
)
async def download_report_file(
    prediction_id: str,
    format: str = Query("pdf", description="Export format: 'pdf', 'json', or 'md'"),
    query_engine: QueryEngine = Depends(get_query_engine),
    exporter: ReportExporter = Depends(get_report_exporter),
    _auth: Optional[str] = Depends(verify_api_key),
) -> FileResponse:
    fmt_clean = format.lower()
    if fmt_clean not in {"pdf", "json", "md"}:
        raise ReportExportException(
            f"Unsupported report format '{format}'. Allowed: 'pdf', 'json', 'md'"
        )

    rec = query_engine.get_by_id(prediction_id)
    if not rec:
        raise PredictionNotFoundException(prediction_id)

    try:
        report = _build_report_from_record(rec)
    except Exception as exc:
        raise ReportExportException(f"Failed to build report data: {exc}") from exc

    out_dir = os.path.join(os.getcwd(), "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)

    file_name = f"report_{prediction_id[:8]}.{fmt_clean}"
    file_path = os.path.join(out_dir, file_name)

    try:
        if fmt_clean == "json":
            exporter.export_json(report, file_path)
            media_type = "application/json"
        elif fmt_clean == "md":
            exporter.export_markdown(report, file_path)
            media_type = "text/markdown; charset=utf-8"
        else:  # pdf
            try:
                exporter.export_pdf(report, file_path)
                media_type = "application/pdf"
            except Exception:
                # Fallback: export as Markdown if PDF renderer unavailable
                file_name = f"report_{prediction_id[:8]}.md"
                file_path = os.path.join(out_dir, file_name)
                exporter.export_markdown(report, file_path)
                media_type = "text/markdown; charset=utf-8"
    except Exception as exc:
        raise ReportExportException(
            f"Failed to render {fmt_clean.upper()} report: {exc}"
        ) from exc

    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type=media_type,
    )
