"""
Medical AI Engine — Visualization Sub-package

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Exports:
    HeatmapRenderer             — Primary facade entry point
    OverlayComposer            — Pipeline master composer
    ColorMapper                — 2D heatmap -> 3D RGB colormaps (inferno, jet, turbo, viridis, magma)
    AlphaBlender               — Image & heatmap transparency blender
    AttentionMaskGenerator     — Binary attention mask thresholding
    BoundingRegionDetector     — Geometric bounding box & polygon detection
    RegionExtractor            — Ranked spatial & intensity region extraction
    VisualizationQualityChecker— Quantitative coverage, sharpness, contrast quality metrics
    ImageExporter              — Disk artifact & metadata JSON exporter
    OverlayConfiguration       — Configuration dataclass
    OverlayMetadata            — Metadata descriptor dataclass
"""

from medical_ai_engine.explainability.visualization.overlay_exceptions import (
    VisualizationException,
    ColorMappingError,
    OverlayBlendingError,
    RegionExtractionError,
    VisualizationQualityError,
    ExportError,
)
from medical_ai_engine.explainability.visualization.overlay_configuration import (
    OverlayConfiguration,
    SUPPORTED_COLORMAPS,
)
from medical_ai_engine.explainability.visualization.overlay_metadata import (
    OverlayMetadata,
)
from medical_ai_engine.explainability.visualization.color_mapper import (
    ColorMapper,
)
from medical_ai_engine.explainability.visualization.alpha_blender import (
    AlphaBlender,
)
from medical_ai_engine.explainability.visualization.attention_mask_generator import (
    AttentionMaskGenerator,
)
from medical_ai_engine.explainability.visualization.bounding_region_detector import (
    BoundingRegionDetector,
)
from medical_ai_engine.explainability.visualization.region_extractor import (
    RegionExtractor,
)
from medical_ai_engine.explainability.visualization.visualization_quality_checker import (
    VisualizationQualityChecker,
)
from medical_ai_engine.explainability.visualization.overlay_composer import (
    OverlayComposer,
)
from medical_ai_engine.explainability.visualization.image_exporter import (
    ImageExporter,
)
from medical_ai_engine.explainability.visualization.heatmap_renderer import (
    HeatmapRenderer,
)

__all__ = [
    "HeatmapRenderer",
    "OverlayComposer",
    "ColorMapper",
    "AlphaBlender",
    "AttentionMaskGenerator",
    "BoundingRegionDetector",
    "RegionExtractor",
    "VisualizationQualityChecker",
    "ImageExporter",
    "OverlayConfiguration",
    "OverlayMetadata",
    "SUPPORTED_COLORMAPS",
    "VisualizationException",
    "ColorMappingError",
    "OverlayBlendingError",
    "RegionExtractionError",
    "VisualizationQualityError",
    "ExportError",
]
