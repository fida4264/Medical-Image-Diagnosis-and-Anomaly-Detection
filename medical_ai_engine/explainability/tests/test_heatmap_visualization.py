"""
Medical AI Engine — Module 15.4 Heatmap Visualization Test Suite

Comprehensive tests for:
  - Color mapping (inferno, jet, turbo, viridis, magma)
  - Alpha blending transparency ratios [0.1, 0.9]
  - Binary attention mask generation & thresholding
  - Bounding region detection & box rendering
  - Region extraction & ranking
  - Visualization quality metrics computation
  - Image artifact disk export (PNG, JPEG, JSON)
  - HeatmapRenderer facade & ExplainabilityContext population
  - Concurrent multi-threaded visualization pipeline execution
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import pytest
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.visualization import (
    HeatmapRenderer,
    OverlayComposer,
    ColorMapper,
    AlphaBlender,
    AttentionMaskGenerator,
    BoundingRegionDetector,
    RegionExtractor,
    VisualizationQualityChecker,
    ImageExporter,
    OverlayConfiguration,
    OverlayMetadata,
    SUPPORTED_COLORMAPS,
    ColorMappingError,
)


@pytest.fixture
def dummy_original_image():
    # 224x224 RGB Image
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[50:150, 50:150] = 128  # Gray square
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_heatmap():
    # 224x224 Normalized Gaussian Heatmap [0.0, 1.0]
    x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d ** 2 / (2.0 * 0.4 ** 2)))
    return (g - g.min()) / (g.max() - g.min())


# ── 1. ColorMapper Tests ──────────────────────────────────────────────────────

class TestColorMapper:
    @pytest.mark.parametrize("cmap_name", SUPPORTED_COLORMAPS)
    def test_supported_colormaps(self, dummy_heatmap, cmap_name):
        mapper = ColorMapper()
        arr, pil_img = mapper.apply_colormap(dummy_heatmap, colormap_name=cmap_name)
        assert arr.shape == (224, 224, 3)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (224, 224)

    def test_invalid_input_shape_raises(self):
        mapper = ColorMapper()
        with pytest.raises(ColorMappingError):
            mapper.apply_colormap(np.ones(10), colormap_name="inferno")


# ── 2. AlphaBlender Tests ─────────────────────────────────────────────────────

class TestAlphaBlender:
    def test_blend_with_different_alphas(self, dummy_original_image, dummy_heatmap):
        mapper = ColorMapper()
        _, heat_pil = mapper.apply_colormap(dummy_heatmap, "inferno")

        blender = AlphaBlender()
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            arr, pil_img = blender.blend(dummy_original_image, heat_pil, alpha=alpha)
            assert arr.shape == (224, 224, 3)
            assert pil_img.size == (224, 224)

    def test_blend_dimension_mismatch_resizes(self, dummy_heatmap):
        blender = AlphaBlender()
        orig = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        heat = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        arr, pil_img = blender.blend(orig, heat, alpha=0.5)
        assert arr.shape == (100, 100, 3)
        assert pil_img.size == (100, 100)


# ── 3. AttentionMaskGenerator Tests ───────────────────────────────────────────

class TestAttentionMaskGenerator:
    def test_mask_generation(self, dummy_heatmap):
        gen = AttentionMaskGenerator()
        arr, pil_img = gen.generate_mask(dummy_heatmap, threshold=0.3)
        assert arr.shape == (224, 224)
        assert pil_img.mode == "L"
        assert set(np.unique(arr)).issubset({0, 255})

    def test_high_threshold_yields_smaller_mask(self, dummy_heatmap):
        gen = AttentionMaskGenerator()
        m1, _ = gen.generate_mask(dummy_heatmap, threshold=0.2)
        m2, _ = gen.generate_mask(dummy_heatmap, threshold=0.8)
        assert np.sum(m1 > 0) > np.sum(m2 > 0)


# ── 4. BoundingRegionDetector Tests ───────────────────────────────────────────

class TestBoundingRegionDetector:
    def test_bounding_box_detection(self, dummy_heatmap):
        mask_gen = AttentionMaskGenerator()
        binary_mask, _ = mask_gen.generate_mask(dummy_heatmap, threshold=0.3)

        detector = BoundingRegionDetector()
        regions = detector.detect_bounding_regions(binary_mask, dummy_heatmap)

        assert len(regions) > 0
        first = regions[0]
        assert "bbox" in first
        assert "centroid" in first
        assert "polygon" in first
        assert len(first["bbox"]) == 4

    def test_draw_bounding_boxes(self, dummy_original_image, dummy_heatmap):
        mask_gen = AttentionMaskGenerator()
        binary_mask, _ = mask_gen.generate_mask(dummy_heatmap, threshold=0.3)

        detector = BoundingRegionDetector()
        regions = detector.detect_bounding_regions(binary_mask, dummy_heatmap)
        annotated = detector.draw_bounding_boxes(dummy_original_image, regions)

        assert isinstance(annotated, Image.Image)
        assert annotated.size == dummy_original_image.size


# ── 5. RegionExtractor Tests ──────────────────────────────────────────────────

class TestRegionExtractor:
    def test_extract_ranked_regions(self, dummy_heatmap):
        mask_gen = AttentionMaskGenerator()
        binary_mask, _ = mask_gen.generate_mask(dummy_heatmap, threshold=0.2)

        extractor = RegionExtractor()
        regions = extractor.extract_regions(dummy_heatmap, binary_mask, threshold=0.2)

        assert len(regions) > 0
        assert regions[0]["rank"] == 1
        assert "peak_activation" in regions[0]
        assert "centroid" in regions[0]


# ── 6. VisualizationQualityChecker Tests ──────────────────────────────────────

class TestVisualizationQualityChecker:
    def test_quality_metrics_calculation(self, dummy_original_image, dummy_heatmap):
        mapper = ColorMapper()
        heat_arr, _ = mapper.apply_colormap(dummy_heatmap, "inferno")
        blender = AlphaBlender()
        over_arr, _ = blender.blend(dummy_original_image, heat_arr, alpha=0.5)
        orig_arr = np.array(dummy_original_image)

        checker = VisualizationQualityChecker()
        metrics = checker.check_quality(orig_arr, dummy_heatmap, over_arr, threshold=0.2)

        assert "coverage_ratio" in metrics
        assert "heatmap_sharpness" in metrics
        assert "activation_contrast" in metrics
        assert "overlay_visibility" in metrics
        assert 0.0 <= metrics["coverage_ratio"] <= 1.0


# ── 7. ImageExporter Tests ────────────────────────────────────────────────────

class TestImageExporter:
    def test_disk_export(self, dummy_original_image, dummy_heatmap):
        composer = OverlayComposer()
        bundle = composer.compose(dummy_original_image, dummy_heatmap)

        exporter = ImageExporter()
        ctx = ExplainabilityContext(request_id="export-test-01", session_id="sess-01")

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = exporter.export(
                composition_bundle=bundle,
                export_dir=tmp_dir,
                filename_prefix="test_artifact",
                export_format="PNG",
                context=ctx,
            )

            assert os.path.exists(paths["overlay_path"])
            assert os.path.exists(paths["heatmap_path"])
            assert os.path.exists(paths["attention_mask_path"])
            assert os.path.exists(paths["metadata_path"])

            # Verify metadata JSON validity
            with open(paths["metadata_path"], "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                assert meta_json["colormap"] == "inferno"

            # Verify Context populated
            assert ctx.overlay_path == paths["overlay_path"]
            assert ctx.heatmap_path == paths["heatmap_path"]


# ── 8. HeatmapRenderer Facade Tests ───────────────────────────────────────────

class TestHeatmapRenderer:
    def test_render_visualization_populates_context(self, dummy_original_image, dummy_heatmap):
        renderer = HeatmapRenderer()
        ctx = ExplainabilityContext(request_id="renderer-test-01", session_id="sess-01")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = renderer.render_visualization(
                image=dummy_original_image,
                normalized_heatmap=dummy_heatmap,
                context=ctx,
                export_dir=tmp_dir,
            )

            assert "overlay_image" in out
            assert "important_regions" in out
            assert "visualization_metrics" in out

            # Context checks
            assert ctx.overlay_image is not None
            assert len(ctx.important_regions) > 0
            assert "visualization_metrics" in ctx.future_extensions
            assert "visualization_render_ms" in ctx.performance_metrics


# ── 9. Concurrent Execution Tests ─────────────────────────────────────────────

class TestConcurrentVisualization:
    def test_concurrent_render_calls(self, dummy_original_image, dummy_heatmap):
        renderer = HeatmapRenderer()

        def worker(idx: int):
            ctx = ExplainabilityContext(request_id=f"concurrent-{idx}", session_id=f"sess-{idx}")
            res = renderer.render_visualization(
                image=dummy_original_image,
                normalized_heatmap=dummy_heatmap,
                context=ctx,
            )
            return res["overlay_image"].size

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        for sz in results:
            assert sz == (224, 224)
