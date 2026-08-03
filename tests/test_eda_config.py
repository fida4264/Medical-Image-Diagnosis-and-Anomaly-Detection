"""
tests/test_eda_config.py
===========================
Unit tests for Module 5 EDA YAML configuration.

Run with:
    pytest tests/test_eda_config.py -v

Author: Medical AI System — Module 5
"""

import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "eda.yaml"


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestConfigExists:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"configs/eda.yaml not found: {CONFIG_PATH}"

    def test_config_is_valid_yaml(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)


class TestTopLevelSections:
    def test_has_general_section(self, config):
        assert "general" in config

    def test_has_figures_section(self, config):
        assert "figures" in config

    def test_has_colors_section(self, config):
        assert "colors" in config

    def test_has_analysis_section(self, config):
        assert "analysis" in config

    def test_has_quality_score_section(self, config):
        assert "quality_score" in config

    def test_has_reports_section(self, config):
        assert "reports" in config

    def test_has_visualizations_section(self, config):
        assert "visualizations" in config


class TestGeneralSection:
    def test_has_output_dir(self, config):
        assert "output_dir" in config["general"]

    def test_has_random_seed(self, config):
        assert "random_seed" in config["general"]

    def test_random_seed_is_int(self, config):
        assert isinstance(config["general"]["random_seed"], int)

    def test_has_sample_mode(self, config):
        assert "sample_mode" in config["general"]

    def test_sample_mode_is_bool(self, config):
        assert isinstance(config["general"]["sample_mode"], bool)

    def test_has_pixel_analysis_sample(self, config):
        assert "pixel_analysis_sample" in config["general"]

    def test_pixel_analysis_sample_positive(self, config):
        assert config["general"]["pixel_analysis_sample"] > 0


class TestFiguresSection:
    def test_has_dpi(self, config):
        assert "dpi" in config["figures"]

    def test_dpi_is_int(self, config):
        assert isinstance(config["figures"]["dpi"], int)

    def test_dpi_reasonable(self, config):
        assert 72 <= config["figures"]["dpi"] <= 600

    def test_has_save_format(self, config):
        assert "save_format" in config["figures"]

    def test_save_format_is_png(self, config):
        assert config["figures"]["save_format"].lower() in {"png", "jpg", "svg"}


class TestColorsSection:
    def test_has_background(self, config):
        assert "background" in config["colors"]

    def test_has_palette(self, config):
        assert "palette" in config["colors"]

    def test_palette_is_list(self, config):
        assert isinstance(config["colors"]["palette"], list)

    def test_palette_has_colors(self, config):
        assert len(config["colors"]["palette"]) >= 4


class TestAnalysisSection:
    def test_has_dataset_subsection(self, config):
        assert "dataset" in config["analysis"]

    def test_has_class_balance_subsection(self, config):
        assert "class_balance" in config["analysis"]

    def test_has_modality_subsection(self, config):
        assert "modality" in config["analysis"]

    def test_has_image_properties_subsection(self, config):
        assert "image_properties" in config["analysis"]

    def test_has_quality_subsection(self, config):
        assert "quality" in config["analysis"]

    def test_rare_class_threshold_positive(self, config):
        cb = config["analysis"]["class_balance"]
        assert cb.get("rare_class_threshold", 1) > 0

    def test_minority_ratio_threshold_valid(self, config):
        cb = config["analysis"]["class_balance"]
        ratio = cb.get("minority_ratio_threshold", 0.1)
        assert 0.0 < ratio < 1.0


class TestQualityScoreWeights:
    def test_weights_sum_to_100(self, config):
        qs = config["quality_score"]
        total = sum([
            qs.get("class_balance_weight", 0),
            qs.get("image_quality_weight", 0),
            qs.get("corruption_weight", 0),
            qs.get("resolution_consistency_weight", 0),
        ])
        assert total == 100, f"Quality score weights must sum to 100, got {total}"


class TestReportsSection:
    def test_generate_pdf_is_bool(self, config):
        assert isinstance(config["reports"].get("generate_pdf", False), bool)

    def test_generate_markdown_is_bool(self, config):
        assert isinstance(config["reports"].get("generate_markdown", False), bool)

    def test_generate_json_is_bool(self, config):
        assert isinstance(config["reports"].get("generate_json", False), bool)


class TestVisualizationsSection:
    EXPECTED_KEYS = [
        "modality_pie",
        "class_bar",
        "class_treemap",
        "resolution_scatter",
        "class_modality_heatmap",
        "quality_dashboard",
        "dataset_overview_dashboard",
        "augmentation_dashboard",
    ]

    def test_all_expected_vis_keys_present(self, config):
        vis = config["visualizations"]
        for key in self.EXPECTED_KEYS:
            assert key in vis, f"visualization key '{key}' missing from config"

    def test_all_vis_values_are_bool(self, config):
        vis = config["visualizations"]
        for k, v in vis.items():
            assert isinstance(v, bool), f"visualizations.{k} should be bool, got {type(v)}"
