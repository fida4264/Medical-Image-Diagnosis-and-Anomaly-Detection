"""
test_reporting.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Comprehensive unit and integration test suite.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from medical_ai_engine.reporting import (
    DEFAULT_DISCLAIMER,
    ClinicalRecommendation,
    ClinicalSummary,
    MedicalReportData,
    RecommendationBuilder,
    ReportConfiguration,
    ReportExporter,
    ReportGenerator,
    ReportTemplateEngine,
    SummaryBuilder,
)
from medical_ai_engine.reporting.reporting_exceptions import (
    ReportGenerationError,
    ReportingConfigurationError,
)


# ── Fixtures & Dummy Objects ──────────────────────────────────────────────────

class DummyPrediction:
    def __init__(
        self,
        disease: str = "Pneumonia",
        confidence: float = 0.942,
        modality: str = "chest_xray",
        request_id: str = "REQ-TEST-100",
        patient_id: str = "PT-8899",
    ):
        self.predicted_disease = disease
        self.confidence = confidence
        self.modality = modality
        self.request_id = request_id
        self.patient_id = patient_id
        self.processing_time_ms = 11.4


class DummyExplainability:
    def __init__(self):
        self.method = "GradCAM"
        self.focus_coverage_pct = 14.8
        self.peak_intensity = 0.91
        self.heatmap_path = "artifacts/gradcam_cxr.png"
        self.overlay_path = "artifacts/overlay_cxr.png"
        self.processing_time_ms = 7.5


class DummyAnomaly:
    def __init__(self, status="KNOWN_DISEASE", risk_level="LOW", risk_score=0.12):
        self.status = status
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.reason = "Sample within training distribution."
        self.recommended_action = "ROUTINE_FOLLOW_UP"
        self.processing_time_ms = 3.2


@pytest.fixture
def generator():
    cfg = ReportConfiguration.fast_test()
    return ReportGenerator(config=cfg)


# ── Test Report Configuration ────────────────────────────────────────────────

class TestReportConfiguration:
    def test_default_configuration(self):
        cfg = ReportConfiguration.default()
        assert cfg.institution_name == "Medical AI Diagnostic & Anomaly Detection Platform"
        assert cfg.include_disclaimer is True
        assert "json" in cfg.export_formats
        assert "pdf" in cfg.export_formats

    def test_invalid_institution_name_raises(self):
        with pytest.raises(ReportingConfigurationError):
            ReportConfiguration(institution_name="   ")

    def test_unsupported_export_format_raises(self):
        with pytest.raises(ReportingConfigurationError):
            ReportConfiguration(export_formats=["json", "docx"])


# ── Test Summary Builder ──────────────────────────────────────────────────────

class TestSummaryBuilder:
    def test_summary_construction(self):
        builder = SummaryBuilder()
        pred = DummyPrediction()
        exp = DummyExplainability()
        anom = DummyAnomaly()

        summary = builder.build_summary(pred, exp, anom)

        assert "Pneumonia" in summary.primary_finding
        assert "94.2%" in summary.primary_finding
        assert "GradCAM" in summary.explainability_narrative
        assert "KNOWN_DISEASE" in summary.anomaly_narrative
        assert len(summary.key_observations) >= 3

    def test_summary_with_missing_optional_components(self):
        builder = SummaryBuilder()
        pred = DummyPrediction(disease="Melanoma", confidence=0.88, modality="skin")

        summary = builder.build_summary(pred, explainability_result=None, anomaly_result=None)

        assert "Melanoma" in summary.primary_finding
        assert "was not requested" in summary.explainability_narrative
        assert "was not executed" in summary.anomaly_narrative


# ── Test Recommendation Builder ──────────────────────────────────────────────

class TestRecommendationBuilder:
    def test_normal_routine_recommendation(self):
        builder = RecommendationBuilder()
        pred = DummyPrediction(disease="Normal", confidence=0.96)
        anom = DummyAnomaly(status="KNOWN_DISEASE", risk_level="LOW")

        rec = builder.build_recommendation(pred, anom)
        assert rec.action_code == "ROUTINE_FOLLOW_UP"
        assert rec.urgency_level == "ROUTINE"

    def test_ood_emergent_recommendation(self):
        builder = RecommendationBuilder()
        pred = DummyPrediction(disease="Glioma", confidence=0.91)
        anom = DummyAnomaly(status="OUT_OF_DISTRIBUTION", risk_level="CRITICAL")

        rec = builder.build_recommendation(pred, anom)
        assert rec.urgency_level == "EMERGENT"
        assert "Triage" in rec.recommended_action or "IMMEDIATE" in rec.action_code

    def test_low_confidence_urgent_recommendation(self):
        builder = RecommendationBuilder()
        pred = DummyPrediction(disease="COVID-19", confidence=0.55)

        rec = builder.build_recommendation(pred, anomaly_result=None)
        assert rec.action_code == "REPEAT_EXAMINATION_OR_SECOND_OPINION"
        assert rec.urgency_level == "URGENT"


# ── Test Report Generator & Exporters ────────────────────────────────────────

class TestReportGenerator:
    def test_generate_report(self, generator):
        pred = DummyPrediction()
        exp = DummyExplainability()
        anom = DummyAnomaly()

        report = generator.generate_report(
            prediction_result=pred,
            explainability_result=exp,
            anomaly_result=anom,
        )

        assert isinstance(report, MedicalReportData)
        assert report.predicted_disease == "Pneumonia"
        assert report.confidence == pytest.approx(0.942)
        assert report.patient_info.patient_id == "PT-8899"
        assert len(report.heatmap_references) >= 1
        assert report.disclaimer == DEFAULT_DISCLAIMER

    def test_generate_report_none_prediction_raises(self, generator):
        with pytest.raises(ReportGenerationError):
            generator.generate_report(prediction_result=None)

    def test_generate_and_export_all_formats(self, generator):
        pred = DummyPrediction()
        exp = DummyExplainability()
        anom = DummyAnomaly()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.generate_and_export(
                prediction_result=pred,
                explainability_result=exp,
                anomaly_result=anom,
                output_directory=tmpdir,
                file_prefix="test_medical_report",
                formats=["json", "markdown", "pdf"],
            )

            assert "json" in paths and paths["json"].exists()
            assert "markdown" in paths and paths["markdown"].exists()
            assert "pdf" in paths and paths["pdf"].exists()

            # Verify JSON valid
            json_text = paths["json"].read_text(encoding="utf-8")
            data = json.loads(json_text)
            assert data["predicted_disease"] == "Pneumonia"
            assert data["confidence_pct"] == 94.2

            # Verify Markdown valid
            md_text = paths["markdown"].read_text(encoding="utf-8")
            assert "# Medical AI Diagnostic & Anomaly Detection Report" in md_text
            assert "Pneumonia" in md_text
            assert "GradCAM" in md_text
            assert "DISCLAIMER" in md_text

            # Verify PDF binary header
            pdf_bytes = paths["pdf"].read_bytes()
            assert pdf_bytes.startswith(b"%PDF-")

    def test_dictionary_input_duck_typing(self, generator):
        dict_pred = {
            "predicted_disease": "Brain Hemorrhage",
            "confidence": 0.915,
            "modality": "brain_ct",
            "request_id": "REQ-DICT-99",
            "patient_id": "PT-DICT-01",
        }

        report = generator.generate_report(prediction_result=dict_pred)
        assert report.predicted_disease == "Brain Hemorrhage"
        assert report.patient_info.modality == "brain_ct"
