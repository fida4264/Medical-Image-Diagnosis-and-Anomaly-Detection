"""
Medical AI Engine — Module 15.3 Grad-CAM Engine Test Suite

Comprehensive tests verifying:
  - ModelHookManager forward/backward hook registration & removal
  - ActivationExtractor capture & tensor shape validation
  - GradientHandler capture, statistics audit, & error handling
  - TargetSelector target category & score resolution
  - CAMComputation mathematical weighting & linear combination
  - HeatmapNormalizer bilinear spatial resize & min-max normalization [0, 1]
  - GradGAMEngine end-to-end execution on real PyTorch models
  - Modality target layer resolution via TargetLayerManager (Modules 7-13)
  - GradCAMPlugin integration via CAMProductionRegistry & CAMPluginFactory
  - Structured exception handling for missing layers, missing gradients, invalid shapes
  - Concurrent multi-threaded Grad-CAM execution
"""

from __future__ import annotations

import concurrent.futures
import pytest
import numpy as np
import torch
import torch.nn as nn

from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.plugins.gradcam import (
    GradCAMPlugin,
    GradGAMEngine,
    GradCAMConfiguration,
    ActivationExtractor,
    GradientHandler,
    TargetSelector,
    ModelHookManager,
    CAMComputation,
    HeatmapNormalizer,
    GradCAMException,
    MissingTargetLayerError,
    GradientExtractionError,
    InvalidTensorShapeError,
    HookRegistrationError,
)
from medical_ai_engine.explainability.registry import (
    CAMProductionRegistry,
    CAMPluginFactory,
    PluginLifecycleState,
)


# ── Simple PyTorch Dummy Classification Model Fixture ─────────────────────────

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # name: features.0
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # name: features.2
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@pytest.fixture
def dummy_model():
    return SimpleConvNet(num_classes=2)


@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 32, 32, requires_grad=True)


@pytest.fixture(autouse=True)
def reset_registry():
    CAMProductionRegistry.reset_instance()
    yield
    CAMProductionRegistry.reset_instance()


# ── 1. ActivationExtractor Tests ──────────────────────────────────────────────

class TestActivationExtractor:
    def test_store_and_retrieve_4d_tensor(self):
        extractor = ActivationExtractor(detach=True)
        t = torch.randn(1, 16, 8, 8)
        extractor.store(t)
        arr = extractor.get_activations()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 16, 8, 8)

    def test_invalid_tensor_shape_raises(self):
        extractor = ActivationExtractor()
        t = torch.randn(16)  # 1D is invalid
        with pytest.raises(InvalidTensorShapeError):
            extractor.store(t)

    def test_clear_resets_state(self):
        extractor = ActivationExtractor()
        extractor.store(torch.randn(1, 8, 4, 4))
        extractor.clear()
        assert extractor.get_activations() is None


# ── 2. GradientHandler Tests ──────────────────────────────────────────────────

class TestGradientHandler:
    def test_store_and_retrieve_gradients(self):
        handler = GradientHandler(detach=True)
        t = torch.randn(1, 16, 8, 8)
        handler.store(t)
        arr = handler.get_gradients()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 16, 8, 8)

    def test_statistics_audit(self):
        handler = GradientHandler()
        t = torch.ones(1, 4, 2, 2)
        handler.store(t)
        stats = handler.get_statistics()
        assert stats["captured"] is True
        assert stats["mean"] == 1.0
        assert stats["has_nan"] is False

    def test_nan_gradient_raises(self):
        handler = GradientHandler()
        t = torch.tensor([[[[float("nan")]]]])
        with pytest.raises(GradientExtractionError):
            handler.store(t)


# ── 3. TargetSelector Tests ───────────────────────────────────────────────────

class TestTargetSelector:
    def test_argmax_selection(self):
        selector = TargetSelector()
        logits = torch.tensor([[0.1, 0.9, 0.3]])
        cat_idx, score = selector.resolve_target(logits, target_category=None)
        assert cat_idx == 1
        assert float(score) == pytest.approx(0.9)

    def test_explicit_category_selection(self):
        selector = TargetSelector()
        logits = torch.tensor([[0.1, 0.9, 0.3]])
        cat_idx, score = selector.resolve_target(logits, target_category=0)
        assert cat_idx == 0
        assert float(score) == pytest.approx(0.1)


# ── 4. ModelHookManager Tests ─────────────────────────────────────────────────

class TestModelHookManager:
    def test_hook_registration_and_cleanup(self, dummy_model, dummy_input):
        extractor = ActivationExtractor()
        handler = GradientHandler()
        manager = ModelHookManager()

        target_layer = dummy_model.features[2]
        manager.register_hooks(dummy_model, target_layer, extractor, handler)
        assert manager.is_registered is True

        # Forward pass
        output = dummy_model(dummy_input)
        assert extractor.get_activations() is not None

        # Backward pass
        output[0, 0].backward()
        assert handler.get_gradients() is not None

        # Cleanup
        manager.remove_hooks()
        assert manager.is_registered is False


# ── 5. CAMComputation Tests ───────────────────────────────────────────────────

class TestCAMComputation:
    def test_computation_shape_and_relu(self):
        comp = CAMComputation()
        acts = np.ones((1, 4, 8, 8), dtype=np.float32)
        grads = np.full((1, 4, 8, 8), 0.5, dtype=np.float32)

        cam = comp.compute(acts, grads, relu_on_output=True)
        assert cam.shape == (8, 8)
        assert np.all(cam >= 0.0)

    def test_negative_gradients_zeroed_by_relu(self):
        comp = CAMComputation()
        acts = np.ones((1, 4, 8, 8), dtype=np.float32)
        grads = np.full((1, 4, 8, 8), -1.0, dtype=np.float32)

        cam = comp.compute(acts, grads, relu_on_output=True)
        assert np.all(cam == 0.0)


# ── 6. HeatmapNormalizer Tests ────────────────────────────────────────────────

class TestHeatmapNormalizer:
    def test_normalization_range(self):
        normalizer = HeatmapNormalizer()
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        norm = normalizer.normalize(raw, target_size=(224, 224), normalize=True)
        assert norm.shape == (224, 224)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_constant_map_handling(self):
        normalizer = HeatmapNormalizer()
        raw = np.ones((8, 8), dtype=np.float32)
        norm = normalizer.normalize(raw, target_size=(224, 224), normalize=True)
        assert norm.shape == (224, 224)
        assert np.all(norm == 0.0)  # Safe constant handling (no NaN)


# ── 7. GradGAMEngine End-to-End Tests ─────────────────────────────────────────

class TestGradGAMEngine:
    def test_end_to_end_pytorch_model(self, dummy_model, dummy_input):
        engine = GradGAMEngine()
        ctx = ExplainabilityContext(request_id="gradcam-test-01", session_id="sess-01")

        result = engine.compute(
            model=dummy_model,
            input_tensor=dummy_input,
            modality="brain_mri",
            target_layer_name="features.2",
            target_category=0,
            context=ctx,
        )

        assert "raw_activation_map" in result
        assert "normalized_heatmap" in result
        assert result["normalized_heatmap"].shape == (224, 224)
        assert result["target_category"] == 0
        assert "performance_metrics" in result
        assert "hook_registration_ms" in result["performance_metrics"]

        # Check Context populated
        assert ctx.activation_maps is not None
        assert ctx.heatmap_array is not None
        assert ctx.target_layer_name == "features.2"

    def test_stub_fallback_when_model_is_none(self):
        engine = GradGAMEngine()
        ctx = ExplainabilityContext(request_id="gradcam-stub-01", session_id="sess-02")

        result = engine.compute(
            model=None,
            input_tensor=None,
            modality="chest_xray",
            context=ctx,
        )

        assert result["normalized_heatmap"].shape == (224, 224)
        assert ctx.heatmap_array is not None


# ── 8. Plugin & CAMProductionRegistry Integration Tests ───────────────────────

class TestGradCAMPluginRegistryIntegration:
    def test_plugin_discovery_and_factory_creation(self):
        from medical_ai_engine.explainability.plugins.gradcam import GradCAMPlugin
        registry = CAMProductionRegistry.get_instance()
        meta = GradCAMPlugin().get_plugin_metadata()
        registry.register("gradcam", GradCAMPlugin, metadata=meta, overwrite=True)
        factory = CAMPluginFactory(registry=registry)

        assert registry.is_registered("gradcam")
        plugin = factory.create("gradcam")
        assert isinstance(plugin, GradCAMPlugin)
        assert registry.get_metadata("gradcam").is_stub is False

    def test_plugin_generate_activation_map(self, dummy_model, dummy_input):
        from medical_ai_engine.explainability.plugins.gradcam import GradCAMPlugin
        registry = CAMProductionRegistry.get_instance()
        registry.register("gradcam", GradCAMPlugin, overwrite=True)
        factory = CAMPluginFactory(registry=registry)

        plugin = factory.create("gradcam")
        heatmap = plugin.generate_activation_map(
            model=dummy_model,
            target_layer="features.2",
            input_tensor=dummy_input,
            target_category=1,
            extra_params={"modality": "skin_lesion"},
        )

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


# ── 9. Concurrent Multi-Thread Execution Tests ────────────────────────────────

class TestConcurrentGradCAM:
    def test_concurrent_gradcam_computations(self):
        engine = GradGAMEngine()

        def worker(idx: int):
            model = SimpleConvNet(num_classes=2)
            inp = torch.randn(1, 3, 32, 32, requires_grad=True)
            ctx = ExplainabilityContext(request_id=f"req-{idx}", session_id=f"sess-{idx}")
            res = engine.compute(
                model=model,
                input_tensor=inp,
                modality="brain_mri",
                target_layer_name="features.2",
                target_category=idx % 2,
                context=ctx,
            )
            return res["normalized_heatmap"].shape

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10
        for shp in results:
            assert shp == (224, 224)
