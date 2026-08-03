"""
tests/test_eda_visualizations.py
==================================
Unit tests for Module 5 EDA visualization classes.

Tests ChartGenerator, HeatmapGenerator, and DashboardGenerator by:
- Verifying that generated PNG files are created at the expected paths.
- Checking that all public chart methods are callable without raising.
- Testing graceful handling of empty or edge-case data.

NO display server required — matplotlib uses the "Agg" backend.

Run with:
    pytest tests/test_eda_visualizations.py -v

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from eda.visualizations.charts import ChartGenerator
from eda.visualizations.heatmaps import HeatmapGenerator
from eda.visualizations.dashboard import DashboardGenerator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def charts(out_dir: Path) -> ChartGenerator:
    return ChartGenerator(output_dir=out_dir, dpi=72)


@pytest.fixture
def heatmaps(out_dir: Path) -> HeatmapGenerator:
    return HeatmapGenerator(output_dir=out_dir, dpi=72)


@pytest.fixture
def dashboards(out_dir: Path) -> DashboardGenerator:
    return DashboardGenerator(output_dir=out_dir, dpi=72)


@pytest.fixture
def sample_counts() -> dict:
    return {"Class A": 500, "Class B": 200, "Class C": 50, "Class D": 800}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "width":  rng.integers(100, 600, size=100),
        "height": rng.integers(100, 600, size=100),
        "file_size_bytes": rng.integers(50_000, 500_000, size=100),
        "channels": [3] * 100,
        "modality": (["brain_mri"] * 40 + ["chest_xray"] * 40 + ["skin"] * 20),
        "normalized_class": (["glioma"] * 20 + ["no_tumor"] * 20 +
                             ["pneumonia"] * 20 + ["normal"] * 20 +
                             ["melanoma"] * 10 + ["nevus"] * 10),
    })


# ---------------------------------------------------------------------------
# ChartGenerator
# ---------------------------------------------------------------------------

class TestChartGeneratorPie:
    def test_pie_creates_file(self, charts, out_dir, sample_counts):
        result = charts.pie(sample_counts, "Test Pie", "test_pie")
        assert result == out_dir / "test_pie.png"
        assert result.exists()

    def test_pie_too_many_slices(self, charts, out_dir):
        counts = {f"Class_{i}": i * 10 + 1 for i in range(20)}
        result = charts.pie(counts, "Many Slices", "pie_many_slices", max_slices=5)
        assert result.exists()


class TestChartGeneratorBarH:
    def test_bar_h_creates_file(self, charts, out_dir, sample_counts):
        result = charts.bar_h(sample_counts, "H Bar Test", "test_bar_h")
        assert result.exists()

    def test_bar_h_single_item(self, charts, out_dir):
        result = charts.bar_h({"Only": 42}, "Single Item", "bar_h_single")
        assert result.exists()


class TestChartGeneratorBarV:
    def test_bar_v_creates_file(self, charts, out_dir, sample_counts):
        result = charts.bar_v(sample_counts, "V Bar Test", "test_bar_v")
        assert result.exists()


class TestChartGeneratorHistogram:
    def test_histogram_creates_file(self, charts, out_dir):
        values = list(range(100))
        result = charts.histogram(values, "Histogram Test", "test_histogram", bins=20)
        assert result.exists()

    def test_histogram_empty_array(self, charts, out_dir):
        # Should not crash
        result = charts.histogram([], "Empty Histogram", "test_histogram_empty")
        # Returns path (file may not exist if no data)
        assert isinstance(result, Path)

    def test_histogram_single_value(self, charts, out_dir):
        result = charts.histogram([42.0], "Single Value", "histogram_single")
        assert isinstance(result, Path)


class TestChartGeneratorScatter:
    def test_scatter_creates_file(self, charts, out_dir):
        x = list(range(50))
        y = [xi * 2 + np.random.randn() for xi in x]
        result = charts.scatter(x, y, "Scatter Test", "test_scatter",
                                xlabel="X", ylabel="Y")
        assert result.exists()

    def test_scatter_with_labels(self, charts, out_dir):
        x = list(range(30))
        y = list(range(30))
        labels = ["mod_a"] * 15 + ["mod_b"] * 15
        result = charts.scatter(x, y, "Scatter with Labels", "scatter_labels",
                                labels=labels)
        assert result.exists()


class TestChartGeneratorViolin:
    def test_violin_creates_file(self, charts, out_dir):
        data = {
            "brain_mri":  [100, 200, 150, 300, 250],
            "chest_xray": [50, 80, 120, 90],
        }
        result = charts.violin(data, "Violin Test", "test_violin")
        assert result.exists()

    def test_violin_empty_data(self, charts, out_dir):
        result = charts.violin({}, "Empty Violin", "violin_empty")
        assert isinstance(result, Path)


class TestChartGeneratorBox:
    def test_box_creates_file(self, charts, out_dir):
        data = {"brain_mri": [100, 200, 300], "skin": [50, 60, 70, 80]}
        result = charts.box(data, "Box Test", "test_box")
        assert result.exists()


class TestChartGeneratorTreemap:
    def test_treemap_fallback_to_bar(self, charts, out_dir, sample_counts):
        # squarify may or may not be installed — should succeed either way
        result = charts.treemap(sample_counts, "Treemap Test", "test_treemap")
        assert isinstance(result, Path)


class TestChartGeneratorLine:
    def test_line_creates_file(self, charts, out_dir):
        x = list(range(10))
        y = [xi ** 2 for xi in x]
        result = charts.line(x, y, "Line Test", "test_line")
        assert result.exists()


class TestChartGeneratorStackedBar:
    def test_stacked_bar_creates_file(self, charts, out_dir):
        data = {
            "brain_mri":  {"train": 100, "val": 20, "test": 30},
            "chest_xray": {"train": 200, "val": 40, "test": 60},
        }
        result = charts.stacked_bar(data, "Stacked Bar Test", "test_stacked_bar")
        assert result.exists()


class TestChartGeneratorGroupedBar:
    def test_grouped_bar_creates_file(self, charts, out_dir):
        data = {
            "brain_mri":  {"glioma": 100, "no_tumor": 80},
            "chest_xray": {"pneumonia": 200, "normal": 150},
        }
        result = charts.grouped_bar(data, "Grouped Bar Test", "test_grouped_bar")
        assert result.exists()


# ---------------------------------------------------------------------------
# HeatmapGenerator
# ---------------------------------------------------------------------------

class TestHeatmapGeneratorCorrelation:
    def test_correlation_creates_file(self, heatmaps, out_dir, sample_df):
        result = heatmaps.correlation_heatmap(
            sample_df,
            title="Correlation Test",
            filename="test_correlation",
            columns=["width", "height", "file_size_bytes"],
        )
        assert result.exists()

    def test_correlation_insufficient_columns(self, heatmaps, out_dir):
        df = pd.DataFrame({"only_one": [1, 2, 3]})
        result = heatmaps.correlation_heatmap(df, "Insufficient", "corr_insufficient")
        assert isinstance(result, Path)


class TestHeatmapGeneratorClassModality:
    def test_class_modality_creates_file(self, heatmaps, out_dir, sample_df):
        result = heatmaps.class_modality_heatmap(
            sample_df,
            class_col="normalized_class",
            modality_col="modality",
            title="Class × Modality",
            filename="test_class_modality",
        )
        assert result.exists()

    def test_class_modality_missing_columns(self, heatmaps, out_dir):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = heatmaps.class_modality_heatmap(
            df, "class", "modality", "Missing Cols", "class_modality_missing"
        )
        assert isinstance(result, Path)


class TestHeatmapGeneratorResolutionDensity:
    def test_density_creates_file(self, heatmaps, out_dir):
        widths  = list(range(100, 600, 5))
        heights = list(range(100, 600, 5))
        result = heatmaps.resolution_density_heatmap(
            widths, heights, "Density Test", "test_density"
        )
        assert result.exists()

    def test_density_insufficient_data(self, heatmaps, out_dir):
        result = heatmaps.resolution_density_heatmap(
            [100], [100], "Single Point", "density_single"
        )
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# DashboardGenerator
# ---------------------------------------------------------------------------

class TestDashboardGeneratorDatasetOverview:
    def test_creates_file(self, dashboards, out_dir):
        dataset_stats = {
            "total_images":          300,
            "total_modalities":      3,
            "total_disease_classes": 8,
            "total_dataset_sources": 2,
            "images_per_modality":   {"brain_mri": 100, "chest_xray": 100, "skin": 100},
            "modality_display_names": {},
            "images_per_class":      {"glioma": 50, "normal": 100},
            "split_distribution":    {"train": 200, "val": 50, "test": 50},
        }
        class_stats = {
            "gini_coefficient": 0.25,
            "balance_label": "Slightly Imbalanced",
            "class_counts": {"glioma": 50, "normal": 100, "melanoma": 60},
        }
        modality_stats = {"per_modality": {}, "total_modalities": 3}

        result = dashboards.dataset_overview(
            dataset_stats, class_stats, modality_stats,
            filename="test_overview_dashboard",
        )
        assert result.exists()


class TestDashboardGeneratorQuality:
    def test_creates_file(self, dashboards, out_dir):
        quality_stats = {
            "quality_score": {
                "total_score":                   78.5,
                "grade":                         "B (Good)",
                "class_balance_score":           20.0,
                "image_quality_score":           22.5,
                "corruption_score":              25.0,
                "resolution_consistency_score":  11.0,
            },
            "flag_breakdown": {
                "dark_image": {"count": 10, "pct": 3.3},
                "blurry":     {"count": 5,  "pct": 1.7},
            },
            "per_modality_quality": {
                "brain_mri":  {"flag_rate_pct": 2.5},
                "chest_xray": {"flag_rate_pct": 5.0},
            },
        }
        result = dashboards.quality_dashboard(quality_stats, filename="test_quality_dashboard")
        assert result.exists()

    def test_empty_quality_stats(self, dashboards, out_dir):
        result = dashboards.quality_dashboard({}, filename="quality_empty")
        assert isinstance(result, Path)


class TestDashboardGeneratorAugmentation:
    def test_creates_file(self, dashboards, out_dir):
        aug_stats = {
            "class_balance_improvement": {
                "available":          True,
                "gini_before":        0.35,
                "gini_after":         0.18,
                "improvement":        0.17,
                "class_counts_before": {"glioma": 100, "no_tumor": 300, "meningioma": 50},
                "class_counts_after":  {"glioma": 400, "no_tumor": 600, "meningioma": 200},
            },
            "transformation_frequency": {
                "rotate": 1000, "flip": 800, "brightness": 600,
            },
            "per_modality": {
                "brain_mri":  {"avg_aug_factor": 4.0},
                "chest_xray": {"avg_aug_factor": 3.0},
            },
        }
        result = dashboards.augmentation_dashboard(aug_stats, filename="test_aug_dashboard")
        assert result.exists()

    def test_no_augmentation_data(self, dashboards, out_dir):
        aug_stats = {"class_balance_improvement": {"available": False}}
        result = dashboards.augmentation_dashboard(aug_stats, filename="aug_dashboard_empty")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------

class TestOutputDirectoryCreation:
    def test_chart_generator_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "charts"
        cg = ChartGenerator(output_dir=new_dir, dpi=72)
        assert new_dir.exists()

    def test_heatmap_generator_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "heat_out"
        HeatmapGenerator(output_dir=new_dir, dpi=72)
        assert new_dir.exists()

    def test_dashboard_generator_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "dash_out"
        DashboardGenerator(output_dir=new_dir, dpi=72)
        assert new_dir.exists()
