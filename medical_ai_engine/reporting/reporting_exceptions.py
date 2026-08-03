"""
reporting_exceptions.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Typed exception hierarchy for the reporting subsystem.
"""

from __future__ import annotations


class ReportingException(Exception):
    """Base exception for all reporting subsystem errors."""


class ReportGenerationError(ReportingException):
    """Raised when report construction from prediction/anomaly results fails."""


class TemplateRenderingError(ReportingException):
    """Raised when template rendering (Markdown/JSON/HTML) fails."""


class ReportExportError(ReportingException):
    """Raised when exporting report to JSON, Markdown, or PDF fails."""


class InvalidReportDataError(ReportingException):
    """Raised when input prediction result or report data is invalid/malformed."""


class ReportingConfigurationError(ReportingException):
    """Raised when reporting configuration settings are invalid."""


__all__ = [
    "ReportingException",
    "ReportGenerationError",
    "TemplateRenderingError",
    "ReportExportError",
    "InvalidReportDataError",
    "ReportingConfigurationError",
]
