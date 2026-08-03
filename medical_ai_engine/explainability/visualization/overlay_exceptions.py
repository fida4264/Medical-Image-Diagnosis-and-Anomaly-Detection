"""
Medical AI Engine — Visualization Exceptions

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Typed exception hierarchy for color mapping, alpha blending, region extraction,
quality evaluation, and image export operations.
"""

from __future__ import annotations

from medical_ai_engine.explainability.registry.cam_exceptions import CAMException


class VisualizationException(CAMException):
    """Base exception for all visualization and overlay pipeline errors."""

    def __init__(self, message: str, plugin_key: str = "visualization") -> None:
        super().__init__(message=message, plugin_key=plugin_key)


class ColorMappingError(VisualizationException):
    """Raised when color mapping fails or an unsupported colormap is requested."""

    def __init__(self, colormap_name: str, reason: str = "") -> None:
        msg = f"Color mapping failed for colormap '{colormap_name}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)
        self.colormap_name = colormap_name


class OverlayBlendingError(VisualizationException):
    """Raised when alpha blending of original image and heatmap fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Alpha blending failed: {reason}")


class RegionExtractionError(VisualizationException):
    """Raised during connected component analysis or bounding region extraction."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Region extraction failure: {reason}")


class VisualizationQualityError(VisualizationException):
    """Raised when visualization quality check encounters invalid input or computation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Quality assessment error: {reason}")


class ExportError(VisualizationException):
    """Raised when exporting image artifacts or metadata to disk fails."""

    def __init__(self, filepath: str, reason: str) -> None:
        super().__init__(f"Failed to export visualization artifact to '{filepath}': {reason}")
        self.filepath = filepath
