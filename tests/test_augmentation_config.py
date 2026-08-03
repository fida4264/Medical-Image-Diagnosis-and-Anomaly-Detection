"""
tests/test_augmentation_config.py
====================================
Unit tests for augmentation YAML configuration loading and structure.

Run with:
    pytest tests/test_augmentation_config.py -v

Author: Medical AI System — Module 4
"""

import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation.yaml"

EXPECTED_MODALITIES = [
    "brain_mri", "brain_ct", "chest_xray",
    "retina", "skin", "ultrasound", "histopathology",
]


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestConfigExists:
    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"Config file not found: {CONFIG_PATH}"

    def test_config_is_valid_yaml(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict)


class TestConfigTopLevelKeys:
    def test_has_general_section(self, config):
        assert "general" in config

    def test_has_common_section(self, config):
        assert "common" in config

    def test_has_modalities_section(self, config):
        assert "modalities" in config

    def test_has_visualization_section(self, config):
        assert "visualization" in config

    def test_has_class_balancing_section(self, config):
        assert "class_balancing" in config


class TestGeneralSection:
    def test_has_random_seed(self, config):
        assert "random_seed" in config["general"]

    def test_has_num_workers(self, config):
        assert "num_workers" in config["general"]

    def test_has_augmentations_per_image(self, config):
        assert "augmentations_per_image" in config["general"]

    def test_seed_is_int(self, config):
        assert isinstance(config["general"]["random_seed"], int)

    def test_workers_positive(self, config):
        assert config["general"]["num_workers"] > 0

    def test_augmentations_per_image_positive(self, config):
        assert config["general"]["augmentations_per_image"] > 0


class TestModalitiesSection:
    def test_all_modalities_present(self, config):
        modalities = config.get("modalities", {})
        for mod in EXPECTED_MODALITIES:
            assert mod in modalities, f"Modality '{mod}' missing from config"

    def test_each_modality_has_rotation(self, config):
        for mod in EXPECTED_MODALITIES:
            mod_cfg = config["modalities"].get(mod, {})
            # rotation section is optional per-modality (falls back to common)
            if "rotation" in mod_cfg:
                assert "limit" in mod_cfg["rotation"]

    def test_no_modality_enables_vertical_flip_for_brain(self, config):
        for mod in ["brain_mri", "brain_ct"]:
            vflip = config["modalities"].get(mod, {}).get("vertical_flip", {})
            enabled = vflip.get("enabled", False)
            assert not enabled, f"{mod} must not enable vertical flip"

    def test_no_modality_enables_vertical_flip_for_xray(self, config):
        vflip = config["modalities"].get("chest_xray", {}).get("vertical_flip", {})
        enabled = vflip.get("enabled", False)
        assert not enabled

    def test_skin_enables_flips(self, config):
        skin = config["modalities"].get("skin", {})
        assert skin.get("horizontal_flip", {}).get("enabled", False) is True
        assert skin.get("vertical_flip", {}).get("enabled", False) is True

    def test_histopathology_enables_flips(self, config):
        histo = config["modalities"].get("histopathology", {})
        assert histo.get("horizontal_flip", {}).get("enabled", False) is True


class TestClassBalancingSection:
    def test_has_enabled_key(self, config):
        assert "enabled" in config["class_balancing"]

    def test_has_strategy_key(self, config):
        assert "strategy" in config["class_balancing"]

    def test_strategy_is_valid(self, config):
        valid = {"minority_only", "all_classes", "target_count"}
        assert config["class_balancing"]["strategy"] in valid
