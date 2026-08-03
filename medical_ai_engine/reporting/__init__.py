"""
__init__.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Package public API for reporting subsystem.
"""

from __future__ import annotations

from medical_ai_engine.reporting.recommendation_builder import (
    ClinicalRecommendation,
    RecommendationBuilder,
)
from medical_ai_engine.reporting.report_configuration import (
    DEFAULT_DISCLAIMER,
    ReportConfiguration,
)
from medical_ai_engine.reporting.report_exporter import ReportExporter
from medical_ai_engine.reporting.report_generator import ReportGenerator
from medical_ai_engine.reporting.report_templates import (
    MedicalReportData,
    PatientInfo,
    ProcessingStats,
    ReportTemplateEngine,
)
from medical_ai_engine.reporting.reporting_exceptions import (
    InvalidReportDataError,
    ReportExportError,
    ReportGenerationError,
    ReportingConfigurationError,
    ReportingException,
    TemplateRenderingError,
)
from medical_ai_engine.reporting.summary_builder import ClinicalSummary, SummaryBuilder

__all__ = [
    # Facade & Main Engine
    "ReportGenerator",
    "ReportExporter",
    "ReportTemplateEngine",
    # Builders
    "SummaryBuilder",
    "RecommendationBuilder",
    # Data Models
    "MedicalReportData",
    "PatientInfo",
    "ProcessingStats",
    "ClinicalSummary",
    "ClinicalRecommendation",
    # Configuration
    "ReportConfiguration",
    "DEFAULT_DISCLAIMER",
    # Exceptions
    "ReportingException",
    "ReportGenerationError",
    "TemplateRenderingError",
    "ReportExportError",
    "InvalidReportDataError",
    "ReportingConfigurationError",
]
