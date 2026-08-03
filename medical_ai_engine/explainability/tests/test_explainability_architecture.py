"""
Medical AI Engine — Module 15.1 Architecture Test Suite

Tests verifying the Explainability Framework Architecture contracts,
plugin registry, dataclasses, and pipeline orchestration stubs.
"""

from __future__ import annotations

import pytest
import numpy as np
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration
    return ExplainabilityConfiguration.default()


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


@pytest.fixture
def dummy_activation():
    x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
    g = np.exp(-(x**2 + y**2) / (2 * 0.5**2))
    return (g - g.min()) / (g.max() - g.min())


@pytest.fixture(autouse=True)
def reset_cam_registry():
    from medical_ai_engine.explainability.cam_registry import CAMRegistry
    CAMRegistry.reset_instance()
    yield
    CAMRegistry.reset_instance()


# ── Configuration Tests ───────────────────────────────────────────────────────

class TestExplainabilityConfiguration:
    def test_default_config_has_expected_method(self, config):
        assert config.default_cam_method == "gradcam"

    def test_default_config_has_7_modalities(self, config):
        assert len(config.default_target_layers) == 7

    def test_config_target_layer_resolution(self, config):
        assert config.get_target_layer_for_modality("brain_mri") == "layer4"
        assert config.get_target_layer_for_modality("chest_xray") == "features"
        assert config.get_target_layer_for_modality("unknown_mod") == "layer4"

    def test_config_supported_methods_count(self, config):
        assert len(config.supported_methods) == 7

    def test_config_to_dict(self, config):
        d = config.to_dict()
        assert "default_cam_method" in d
        assert "supported_methods" in d
        assert "colormap" in d

    def test_config_overlay_alpha_in_range(self, config):
        assert 0.0 <= config.overlay_alpha <= 1.0

    def test_config_enabled_by_default(self, config):
        assert config.enabled is True


# ── ExplainabilityContext Tests ───────────────────────────────────────────────

class TestExplainabilityContext:
    def test_context_creation(self):
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        ctx = ExplainabilityContext(request_id="xai-001", session_id="sess-001")
        assert ctx.request_id == "xai-001"
        assert ctx.session_id == "sess-001"
        assert ctx.execution_state == "CREATED"

    def test_context_add_warning(self):
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        ctx = ExplainabilityContext(request_id="xai-002", session_id="sess-002")
        ctx.add_warning("Test warning")
        assert len(ctx.warnings) == 1
        assert ctx.warnings[0] == "Test warning"

    def test_context_add_error(self):
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        ctx = ExplainabilityContext(request_id="xai-003", session_id="sess-003")
        ctx.add_error(stage="Stage4", error="Test error")
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["stage"] == "Stage4"

    def test_context_record_metric(self):
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        ctx = ExplainabilityContext(request_id="xai-004", session_id="sess-004")
        ctx.record_metric("heatmap_ms", 5.123)
        assert ctx.performance_metrics["heatmap_ms"] == 5.123

    def test_context_timestamp_set(self):
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        ctx = ExplainabilityContext(request_id="xai-005", session_id="sess-005")
        assert ctx.timestamp is not None and len(ctx.timestamp) > 0


# ── ExplainabilityResult Tests ────────────────────────────────────────────────

class TestExplainabilityResult:
    def test_result_is_immutable(self):
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(
            request_id="xai-r-001",
            session_id="sess-001",
            method="gradcam",
            predicted_class="tumor",
            confidence=0.91,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.status = "FAILED"  # type: ignore

    def test_result_is_success(self):
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(
            request_id="xai-r-002",
            session_id="sess-002",
            status="SUCCESS",
        )
        assert result.is_success is True

    def test_result_not_success_on_failed(self):
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(
            request_id="xai-r-003",
            session_id="sess-003",
            status="FAILED",
        )
        assert result.is_success is False

    def test_result_to_dict(self):
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(
            request_id="xai-r-004",
            session_id="sess-004",
            method="scorecam",
            predicted_class="normal",
            confidence=0.87,
        )
        d = result.to_dict()
        assert d["method"] == "scorecam"
        assert d["predicted_class"] == "normal"
        assert d["confidence"] == 0.87

    def test_result_to_json_valid(self):
        import json
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(request_id="xai-r-005", session_id="sess-005")
        parsed = json.loads(result.to_json())
        assert "request_id" in parsed


# ── BaseCAMMethod Tests ───────────────────────────────────────────────────────

class TestBaseCAMMethod:
    def test_stub_returns_2d_array(self):
        from medical_ai_engine.explainability.base_cam_method import GradCAMMethodStub
        stub = GradCAMMethodStub()
        arr = stub.generate_activation_map(model=None, target_layer=None, input_tensor=None)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2

    def test_stub_output_normalized(self):
        from medical_ai_engine.explainability.base_cam_method import GradCAMMethodStub
        stub = GradCAMMethodStub()
        arr = stub.generate_activation_map(model=None, target_layer=None, input_tensor=None)
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_all_method_stubs_present(self):
        from medical_ai_engine.explainability.base_cam_method import (
            GradCAMMethodStub, GradCAMPlusPlusMethodStub, LayerCAMMethodStub,
            ScoreCAMMethodStub, EigenCAMMethodStub,
            IntegratedGradientsMethodStub, GuidedBackpropMethodStub,
        )
        stubs = [
            GradCAMMethodStub(), GradCAMPlusPlusMethodStub(), LayerCAMMethodStub(),
            ScoreCAMMethodStub(), EigenCAMMethodStub(),
            IntegratedGradientsMethodStub(), GuidedBackpropMethodStub(),
        ]
        assert len(stubs) == 7

    def test_stub_get_metadata_returns_dict(self):
        from medical_ai_engine.explainability.base_cam_method import GradCAMMethodStub
        stub = GradCAMMethodStub()
        meta = stub.get_metadata()
        assert "method_name" in meta
        assert "requires_gradients" in meta


# ── CAMRegistry Tests ─────────────────────────────────────────────────────────

class TestCAMRegistry:
    def test_registry_singleton(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        r1 = CAMRegistry.get_instance()
        r2 = CAMRegistry.get_instance()
        assert r1 is r2

    def test_registry_has_7_default_methods(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        reg = CAMRegistry.get_instance()
        methods = reg.list_methods()
        assert len(methods) == 7

    def test_registry_has_gradcam(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        reg = CAMRegistry.get_instance()
        assert reg.is_registered("gradcam")

    def test_registry_get_method_class(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod
        reg = CAMRegistry.get_instance()
        cls = reg.get_method_class("gradcam")
        assert issubclass(cls, BaseCAMMethod)

    def test_registry_get_method_instance_cached(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        reg = CAMRegistry.get_instance()
        inst1 = reg.get_method_instance("gradcam")
        inst2 = reg.get_method_instance("gradcam")
        assert inst1 is inst2

    def test_registry_custom_plugin_registration(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod

        class MyCustomCAM(BaseCAMMethod):
            def __init__(self):
                super().__init__(name="my_custom_cam", requires_gradients=False)

        reg = CAMRegistry.get_instance()
        reg.register("my_custom_cam", MyCustomCAM)
        assert reg.is_registered("my_custom_cam")

    def test_registry_unknown_method_raises(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        reg = CAMRegistry.get_instance()
        with pytest.raises(KeyError):
            reg.get_method_class("nonexistent_method")

    def test_registry_get_all_metadata(self):
        from medical_ai_engine.explainability.cam_registry import CAMRegistry
        reg = CAMRegistry.get_instance()
        meta = reg.get_all_metadata()
        assert "gradcam" in meta
        assert "method_name" in meta["gradcam"]


# ── CAMFactory Tests ──────────────────────────────────────────────────────────

class TestCAMFactory:
    def test_factory_creates_gradcam(self):
        from medical_ai_engine.explainability.cam_factory import CAMFactory
        from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod
        factory = CAMFactory()
        method = factory.create_cam_method("gradcam")
        assert isinstance(method, BaseCAMMethod)

    def test_factory_creates_scorecam(self):
        from medical_ai_engine.explainability.cam_factory import CAMFactory
        factory = CAMFactory()
        method = factory.create_cam_method("scorecam")
        assert method.method_name == "scorecam"

    def test_factory_raises_on_unknown_method(self):
        from medical_ai_engine.explainability.cam_factory import CAMFactory
        factory = CAMFactory()
        with pytest.raises(KeyError):
            factory.create_cam_method("nonexistent_xyz")

    def test_factory_default_is_gradcam(self):
        from medical_ai_engine.explainability.cam_factory import CAMFactory
        method = CAMFactory.create_default()
        assert method.method_name == "gradcam"


# ── TargetLayerManager Tests ──────────────────────────────────────────────────

class TestTargetLayerManager:
    def test_resolve_returns_layer_name_when_no_model(self, config):
        from medical_ai_engine.explainability.target_layer_manager import TargetLayerManager
        mgr = TargetLayerManager(config=config)
        result = mgr.resolve_target_layer(model=None, modality="brain_mri")
        assert result == "layer4"

    def test_resolve_respects_explicit_layer(self, config):
        from medical_ai_engine.explainability.target_layer_manager import TargetLayerManager
        mgr = TargetLayerManager(config=config)
        result = mgr.resolve_target_layer(model=None, modality="brain_mri", explicit_layer_name="layer3")
        assert result == "layer3"

    def test_list_candidates_without_model(self, config):
        from medical_ai_engine.explainability.target_layer_manager import TargetLayerManager
        mgr = TargetLayerManager(config=config)
        candidates = mgr.list_candidate_layers(model=None)
        assert isinstance(candidates, list)
        assert len(candidates) > 0


# ── HeatmapGenerator Tests ────────────────────────────────────────────────────

class TestHeatmapGenerator:
    def test_generate_heatmap_returns_array_and_image(self, config, dummy_activation):
        from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
        gen = HeatmapGenerator(config=config)
        arr, pil_img = gen.generate_heatmap(dummy_activation)
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert pil_img.mode == "RGB"

    def test_generate_heatmap_correct_size(self, config, dummy_activation):
        from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
        gen = HeatmapGenerator(config=config)
        arr, pil_img = gen.generate_heatmap(dummy_activation, target_size=(224, 224))
        assert arr.shape[:2] == (224, 224)


# ── OverlayGenerator Tests ────────────────────────────────────────────────────

class TestOverlayGenerator:
    def test_overlay_returns_pil_image(self, config, dummy_image, dummy_activation):
        from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
        from medical_ai_engine.explainability.overlay_generator import OverlayGenerator

        heatmap_gen = HeatmapGenerator(config=config)
        _, heatmap_pil = heatmap_gen.generate_heatmap(dummy_activation)

        overlay_gen = OverlayGenerator(config=config)
        overlay = overlay_gen.generate_overlay(dummy_image, heatmap_pil, alpha=0.5)
        assert overlay.mode == "RGB"

    def test_overlay_correct_size(self, config, dummy_image, dummy_activation):
        from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
        from medical_ai_engine.explainability.overlay_generator import OverlayGenerator

        heatmap_gen = HeatmapGenerator(config=config)
        _, heatmap_pil = heatmap_gen.generate_heatmap(dummy_activation)

        overlay_gen = OverlayGenerator(config=config)
        overlay = overlay_gen.generate_overlay(dummy_image, heatmap_pil)
        assert overlay.size == heatmap_pil.size


# ── ClinicalExplanationBuilder Tests ─────────────────────────────────────────

class TestClinicalExplanationBuilder:
    def test_builder_returns_regions_and_summary(self, config, dummy_activation):
        from medical_ai_engine.explainability.clinical_explanation_builder import ClinicalExplanationBuilder
        builder = ClinicalExplanationBuilder(config=config)
        regions, summary = builder.build_explanation(
            activation_map=dummy_activation,
            modality="brain_mri",
            predicted_disease="tumor",
            confidence=0.92,
        )
        assert isinstance(regions, list)
        assert len(regions) > 0
        assert isinstance(summary, str) and len(summary) > 0

    def test_builder_region_has_required_keys(self, config, dummy_activation):
        from medical_ai_engine.explainability.clinical_explanation_builder import ClinicalExplanationBuilder
        builder = ClinicalExplanationBuilder(config=config)
        regions, _ = builder.build_explanation(dummy_activation, "chest_xray", "pneumonia", 0.85)
        for region in regions:
            assert "rank" in region
            assert "region_name" in region
            assert "activation_score" in region


# ── ExplainabilityJSONBuilder Tests ──────────────────────────────────────────

class TestExplainabilityJSONBuilder:
    def test_json_builder_returns_dict(self):
        import json
        from medical_ai_engine.explainability.explainability_json_builder import ExplainabilityJSONBuilder
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(request_id="xai-j-001", session_id="sess-j-001")
        d = ExplainabilityJSONBuilder.to_dict(result)
        assert isinstance(d, dict)
        assert "request_id" in d

    def test_json_builder_returns_valid_json(self):
        import json
        from medical_ai_engine.explainability.explainability_json_builder import ExplainabilityJSONBuilder
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
        result = ExplainabilityResult(request_id="xai-j-002", session_id="sess-j-002")
        json_str = ExplainabilityJSONBuilder.to_json(result)
        parsed = json.loads(json_str)
        assert parsed["request_id"] == "xai-j-002"


# ── ExplainabilityEngine Integration Tests ────────────────────────────────────

class TestExplainabilityEngine:
    def test_engine_initializes(self, config):
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
        engine = ExplainabilityEngine(config=config)
        assert engine is not None

    def test_engine_supported_methods_count(self, config):
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
        engine = ExplainabilityEngine(config=config)
        methods = engine.get_supported_methods()
        assert len(methods) == 7

    def test_engine_explain_returns_result(self, config, dummy_image):
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
        from medical_ai_engine.explainability.explainability_result import ExplainabilityResult

        engine = ExplainabilityEngine(config=config)
        ctx = ExplainabilityContext(
            request_id="xai-e-001",
            session_id="sess-e-001",
            image=dummy_image,
            modality="brain_mri",
            predicted_class="tumor",
            confidence=0.92,
        )
        result = engine.explain(ctx)
        assert isinstance(result, ExplainabilityResult)
        assert result.status in ("SUCCESS", "FAILED", "SKIPPED")

    def test_engine_disabled_returns_skipped(self, config, dummy_image):
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
        from medical_ai_engine.explainability.explainability_context import ExplainabilityContext

        config.enabled = False
        engine = ExplainabilityEngine(config=config)
        ctx = ExplainabilityContext(request_id="xai-e-002", session_id="sess-e-002")
        result = engine.explain(ctx)
        assert result.status == "SKIPPED"

    def test_engine_plugin_registration(self, config):
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine
        from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod

        class NewPlugin(BaseCAMMethod):
            def __init__(self):
                super().__init__(name="new_plugin", requires_gradients=False)

        engine = ExplainabilityEngine(config=config)
        engine.register_cam_method("new_plugin", NewPlugin)
        assert "new_plugin" in engine.get_supported_methods()


# ── Package Import Smoke Test ─────────────────────────────────────────────────

class TestPackageImports:
    def test_package_imports_cleanly(self):
        import medical_ai_engine.explainability as xai
        assert xai.__version__ == "0.15.1.0"

    def test_all_exports_present(self):
        from medical_ai_engine.explainability import (
            ExplainabilityEngine,
            ExplainabilityContext,
            ExplainabilityResult,
            ExplainabilityConfiguration,
            ExplainabilityPipeline,
            ExplainabilityEventListener,
            CAMRegistry,
            CAMFactory,
            BaseCAMMethod,
            TargetLayerManager,
            HeatmapGenerator,
            OverlayGenerator,
            ClinicalExplanationBuilder,
            ExplainabilityJSONBuilder,
        )
        assert all([
            ExplainabilityEngine, ExplainabilityContext, ExplainabilityResult,
            ExplainabilityConfiguration, ExplainabilityPipeline, ExplainabilityEventListener,
            CAMRegistry, CAMFactory, BaseCAMMethod, TargetLayerManager,
            HeatmapGenerator, OverlayGenerator, ClinicalExplanationBuilder, ExplainabilityJSONBuilder,
        ])

    def test_interface_imports(self):
        from medical_ai_engine.explainability.interfaces.cam_method_interface import ICAMMethod
        from medical_ai_engine.explainability.interfaces.explainability_engine_interface import IExplainabilityEngine
        assert ICAMMethod
        assert IExplainabilityEngine
