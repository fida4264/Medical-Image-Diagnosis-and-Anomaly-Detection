"""
report_exporter.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Multi-format report exporter supporting JSON, Markdown, and PDF.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from medical_ai_engine.reporting.report_templates import MedicalReportData, ReportTemplateEngine
from medical_ai_engine.reporting.reporting_exceptions import ReportExportError

logger = logging.getLogger(__name__)


def _generate_pure_python_pdf(report: MedicalReportData, output_path: Path) -> Path:
    """
    Pure-Python standard PDF-1.4 builder fallback.

    Generates a valid, readable PDF file containing headers, metadata, patient info,
    diagnostic summary, recommendations, and clinical disclaimer without third-party dependencies.
    """
    lines = [
        "MEDICAL AI DIAGNOSTIC & ANOMALY DETECTION REPORT",
        "=" * 55,
        f"Report ID : {report.report_id}",
        f"Created   : {report.created_at}",
        f"Patient ID: {report.patient_info.patient_id}",
        f"Case ID   : {report.patient_info.case_id}",
        f"Modality  : {report.patient_info.modality.replace('_', ' ').title()}",
        "",
        "DIAGNOSTIC SUMMARY",
        "-" * 55,
        f"Predicted Disease : {report.predicted_disease}",
        f"Model Confidence  : {report.confidence * 100:.1f}%",
        f"Primary Finding   : {report.summary.primary_finding}",
        "",
        "DIAGNOSTIC IMPRESSION",
        report.summary.diagnostic_impression,
        "",
        "VISUAL EXPLAINABILITY",
        report.summary.explainability_narrative,
        "",
        "ANOMALY & OOD ASSESSMENT",
        report.summary.anomaly_narrative,
        "",
        "CLINICAL RECOMMENDATIONS",
        "-" * 55,
        f"Action Code       : {report.recommendation.action_code}",
        f"Recommended Action: {report.recommendation.recommended_action}",
        f"Urgency Level     : {report.recommendation.urgency_level}",
        f"Rationale         : {report.recommendation.rationale}",
        "",
        "PROCESSING STATISTICS",
        f"Model Architecture: {report.processing_stats.model_architecture}",
        f"Engine Latency    : {report.processing_stats.total_processing_time_ms:.2f} ms",
        f"Platform Version  : v{report.processing_stats.engine_version}",
        "",
        "MANDATORY AI DISCLAIMER",
        "-" * 55,
        report.disclaimer,
    ]

    text_content = "\n".join(lines)

    # Convert text to standard PDF stream objects
    # Escaping special characters for PDF text literal
    escaped_text = text_content.replace("(", "\\(").replace(")", "\\)")
    text_lines = escaped_text.split("\n")

    # Format text lines into PDF operators
    pdf_text_ops = []
    y_pos = 750
    for line in text_lines[:50]:  # Up to 50 lines on page
        if not line:
            y_pos -= 12
            continue
        pdf_text_ops.append(f"1 0 0 1 50 {y_pos} Tm ({line}) Tj")
        y_pos -= 14

    stream_data = "BT /F1 10 Tf 14 TL " + " ".join(pdf_text_ops) + " ET"
    stream_bytes = stream_data.encode("ascii", errors="replace")
    stream_len = len(stream_bytes)

    # PDF Objects
    obj1 = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5 = f"5 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("ascii") + stream_bytes + b"\nendstream\nendobj\n"

    objects = [obj1, obj2, obj3, obj4, obj5]

    # Calculate xref byte offsets
    offsets = [0]
    curr = 0
    for o in objects:
        offsets.append(curr)
        curr += len(o)

    xref_offset = curr

    # Xref table & Trailer
    xref_lines = [b"xref\n0 6\n0000000000 65535 f \n"]
    for i in range(1, 6):
        # Find position of obj i
        pos = sum(len(objects[j]) for j in range(i - 1))
        xref_lines.append(f"{pos:010d} 00000 n \n".encode("ascii"))

    xref = b"".join(xref_lines)
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")

    pdf_binary = b"".join(objects) + xref + trailer
    output_path.write_bytes(pdf_binary)
    return output_path


class ReportExporter:
    """
    Exports MedicalReportData into JSON, Markdown, and PDF file formats.
    """

    def __init__(self, template_engine: Optional[ReportTemplateEngine] = None) -> None:
        self.template_engine = template_engine or ReportTemplateEngine()

    def export(
        self,
        report: MedicalReportData,
        output_directory: Union[str, Path] = "docs",
        file_prefix: str = "medical_report",
        formats: Optional[list] = None,
    ) -> Dict[str, Path]:
        """
        Export report into requested formats.

        Parameters
        ----------
        report : MedicalReportData
            Report data structure.
        output_directory : str or Path
            Target export directory.
        file_prefix : str
            Base filename prefix (without extension).
        formats : list, optional
            List of formats to export: ['json', 'markdown', 'pdf'].

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping format string to generated Path.
        """
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["json", "markdown", "pdf"]

        exported_files: Dict[str, Path] = {}

        for fmt in formats:
            fmt_clean = fmt.lower().strip()
            if fmt_clean == "json":
                p = out_dir / f"{file_prefix}.json"
                exported_files["json"] = self.export_json(report, p)
            elif fmt_clean in ("markdown", "md"):
                p = out_dir / f"{file_prefix}.md"
                exported_files["markdown"] = self.export_markdown(report, p)
            elif fmt_clean == "pdf":
                p = out_dir / f"{file_prefix}.pdf"
                exported_files["pdf"] = self.export_pdf(report, p)
            else:
                logger.warning("[ReportExporter] Skipped unknown format: %s", fmt)

        return exported_files

    def export_json(self, report: MedicalReportData, output_path: Union[str, Path]) -> Path:
        """Export report to JSON file."""
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report.to_json(), encoding="utf-8")
            logger.info("[ReportExporter] JSON exported: %s", p)
            return p
        except Exception as exc:
            raise ReportExportError(f"Failed to export JSON report: {exc}") from exc

    def export_markdown(self, report: MedicalReportData, output_path: Union[str, Path]) -> Path:
        """Export report to Markdown file."""
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            md_content = self.template_engine.render_markdown(report)
            p.write_text(md_content, encoding="utf-8")
            logger.info("[ReportExporter] Markdown exported: %s", p)
            return p
        except Exception as exc:
            raise ReportExportError(f"Failed to export Markdown report: {exc}") from exc

    def export_pdf(self, report: MedicalReportData, output_path: Union[str, Path]) -> Path:
        """
        Export report to PDF file.

        Tries ReportLab if installed; falls back to pure-Python PDF builder.
        """
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            try:
                import reportlab
                self._export_pdf_reportlab(report, p)
            except ImportError:
                _generate_pure_python_pdf(report, p)

            logger.info("[ReportExporter] PDF exported: %s", p)
            return p
        except Exception as exc:
            raise ReportExportError(f"Failed to export PDF report: {exc}") from exc

    def _export_pdf_reportlab(self, report: MedicalReportData, output_path: Path) -> None:
        """Render PDF using ReportLab Canvas / SimpleDocTemplate if installed."""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"<b>Medical AI Diagnostic Report</b>", styles["Heading1"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Patient ID: {report.patient_info.patient_id}", styles["Normal"]))
        story.append(Paragraph(f"Modality: {report.patient_info.modality}", styles["Normal"]))
        story.append(Paragraph(f"Predicted Disease: {report.predicted_disease}", styles["Heading2"]))
        story.append(Paragraph(f"Confidence: {report.confidence*100:.1f}%", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Summary: {report.summary.diagnostic_impression}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Recommended Action: {report.recommendation.recommended_action}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<i>Disclaimer: {report.disclaimer}</i>", styles["Italic"]))

        doc.build(story)


__all__ = ["ReportExporter"]
