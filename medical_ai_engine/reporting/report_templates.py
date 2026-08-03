"""
report_templates.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Report data model and template rendering engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from medical_ai_engine.reporting.recommendation_builder import ClinicalRecommendation
from medical_ai_engine.reporting.reporting_exceptions import TemplateRenderingError
from medical_ai_engine.reporting.summary_builder import ClinicalSummary


@dataclass
class PatientInfo:
    patient_id: str = "PT-DEMO-001"
    case_id: str = "CASE-2026-0802"
    age: Optional[int] = None
    gender: Optional[str] = None
    modality: str = "chest_xray"
    study_date: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    )


@dataclass
class ProcessingStats:
    inference_latency_ms: float = 12.5
    explainability_latency_ms: float = 8.2
    anomaly_latency_ms: float = 4.1
    total_processing_time_ms: float = 24.8
    model_architecture: str = "DenseNet-121"
    engine_version: str = "0.18.0.0"


@dataclass
class MedicalReportData:
    """
    Complete structured data container for a single Medical AI Report.
    """

    report_id: str
    patient_info: PatientInfo
    predicted_disease: str
    confidence: float
    summary: ClinicalSummary
    recommendation: ClinicalRecommendation
    heatmap_references: List[str] = field(default_factory=list)
    processing_stats: ProcessingStats = field(default_factory=ProcessingStats)
    disclaimer: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire report data structure to JSON-serializable dictionary."""
        d = asdict(self)
        d["confidence_pct"] = round(self.confidence * 100.0 if self.confidence <= 1.0 else self.confidence, 2)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ReportTemplateEngine:
    """
    Renders structured `MedicalReportData` into Markdown text or HTML formats.
    """

    def render_markdown(self, report: MedicalReportData) -> str:
        """Render MedicalReportData into a Markdown report."""
        try:
            p = report.patient_info
            s = report.summary
            r = report.recommendation
            st = report.processing_stats
            conf_pct = report.confidence * 100.0 if report.confidence <= 1.0 else report.confidence

            heatmaps_md = (
                "\n".join(f"- `{ref}`" for ref in report.heatmap_references)
                if report.heatmap_references
                else "- *No visual explainability heatmap references attached.*"
            )

            obs_md = (
                "\n".join(f"- {obs}" for obs in s.key_observations)
                if s.key_observations
                else "- *No specific observations recorded.*"
            )

            md = f"""# Medical AI Diagnostic & Anomaly Detection Report

**Institution:** Medical AI Diagnostic Platform  
**Report ID:** `{report.report_id}`  
**Generated At:** {report.created_at}  

---

## 1. Patient & Case Information

| Attribute | Value |
|-----------|-------|
| **Patient ID** | `{p.patient_id}` |
| **Case ID** | `{p.case_id}` |
| **Image Modality** | {p.modality.replace('_', ' ').title()} |
| **Study Date** | {p.study_date} |

---

## 2. Diagnostic Summary

| Metric | Output |
|--------|--------|
| **Primary Disease Finding** | **{report.predicted_disease}** |
| **Model Confidence** | **{conf_pct:.1f}%** |
| **Primary Finding** | {s.primary_finding} |

### Diagnostic Impression
{s.diagnostic_impression}

### Key Diagnostic Observations
{obs_md}

---

## 3. Visual Explainability (GradCAM / Saliency)

{s.explainability_narrative}

### Heatmap & Overlay Artifact References
{heatmaps_md}

---

## 4. Anomaly & Out-of-Distribution Assessment

{s.anomaly_narrative}

---

## 5. Clinical Recommendations

| Field | Detail |
|-------|--------|
| **Action Code** | `{r.action_code}` |
| **Recommended Action** | **{r.recommended_action}** |
| **Urgency Level** | **{r.urgency_level}** |

### Clinical Rationale
{r.rationale}

### Suggested Follow-Up Actions
{r.suggested_follow_up}

---

## 6. System Processing Statistics

| Parameter | Metric |
|-----------|--------|
| Model Architecture | `{st.model_architecture}` |
| Inference Latency | {st.inference_latency_ms:.2f} ms |
| Explainability Latency | {st.explainability_latency_ms:.2f} ms |
| Anomaly Engine Latency | {st.anomaly_latency_ms:.2f} ms |
| **Total Engine Latency** | **{st.total_processing_time_ms:.2f} ms** |
| Platform Version | `v{st.engine_version}` |

---

## 7. Mandatory AI Disclaimer

> [!IMPORTANT]
> **AI CLINICAL SUPPORT DISCLAIMER**  
> {report.disclaimer}

---
*Report generated by Medical AI Engine v{st.engine_version}*
"""
            return md
        except Exception as exc:
            raise TemplateRenderingError(f"Failed to render Markdown report: {exc}") from exc


__all__ = [
    "PatientInfo",
    "ProcessingStats",
    "MedicalReportData",
    "ReportTemplateEngine",
]
