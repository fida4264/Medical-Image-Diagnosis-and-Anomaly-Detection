"""
Medical AI Engine — Module 15.2 CAM Registry & Plugin System Test Suite

Comprehensive tests for:
  - Dynamic registration & duplicate detection
  - Plugin discovery & filtering (modality, state, framework, gradient-free)
  - Lazy loading & loader isolation
  - Metadata tracking & status check
  - Thread-safe concurrent operations
  - Lifecycle FSM state transitions
  - CAMPluginFactory lifecycle orchestration & creation
  - LRU Caching, TTL expiry, and invalidation
  - Interface validation & custom plugin implementation
"""

from __future__ import annotations

import concurrent.futures
import pytest
import numpy as np

from medical_ai_engine.explainability.plugins.base_cam_plugin import BaseCAMPlugin
from medical_ai_engine.explainability.registry import (
    CAMProductionRegistry,
    CAMPluginFactory,
    CAMPluginLoader,
    CAMPluginCache,
    CAMPluginMetadata,
    CAMPluginConfiguration,
    PluginLifecycleState,
    PluginStatus,
    CAMDuplicateError,
    CAMNotFoundError,
    CAMLifecycleError,
    CAMInvalidPluginError,
)


# ── Dummy Plugin fixture for tests ────────────────────────────────────────────

class DummyTestCAMPlugin(BaseCAMPlugin):
    """Concrete test plugin implementing ICAMPlugin contract."""

    def __init__(self, key: str = "test_cam", version: str = "1.0.0") -> None:
        super().__init__(
            plugin_key=key,
            plugin_version=version,
            requires_gradients_flag=False,
            description="Dummy test plugin for unit testing",
        )
        self.initialize_called = False
        self.ready_called = False
        self.activate_called = False
        self.deactivate_called = False
        self.unload_called = False

    def on_initialize(self) -> None:
        super().on_initialize()
        self.initialize_called = True

    def on_ready(self) -> None:
        super().on_ready()
        self.ready_called = True

    def on_activate(self) -> None:
        super().on_activate()
        self.activate_called = True

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self.deactivate_called = True

    def on_unload(self) -> None:
        super().on_unload()
        self.unload_called = True

    def generate_activation_map(
        self,
        model: str | None = None,
        target_layer: str | None = None,
        input_tensor: str | None = None,
        target_category: int | None = None,
        extra_params: dict | None = None,
    ) -> np.ndarray:
        return np.ones((224, 224), dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_registry():
    CAMProductionRegistry.reset_instance()
    yield
    CAMProductionRegistry.reset_instance()


@pytest.fixture
def registry():
    return CAMProductionRegistry.get_instance()


@pytest.fixture
def factory(registry):
    return CAMPluginFactory(registry=registry)


# ── 1. Registration & Duplicate Detection Tests ──────────────────────────────

class TestRegistration:
    def test_auto_registers_7_builtins(self, registry):
        assert len(registry.list_keys()) == 7
        builtins = ["gradcam", "gradcam_plus_plus", "layercam", "scorecam",
                    "eigencam", "integrated_gradients", "guided_backprop"]
        for b in builtins:
            assert registry.is_registered(b)

    def test_dynamic_registration(self, registry):
        registry.register("dummy_cam", DummyTestCAMPlugin)
        assert registry.is_registered("dummy_cam")
        assert registry.get_plugin_class("dummy_cam") is DummyTestCAMPlugin

    def test_duplicate_registration_raises(self, registry):
        registry.register("dummy_cam", DummyTestCAMPlugin)
        with pytest.raises(CAMDuplicateError):
            registry.register("dummy_cam", DummyTestCAMPlugin, overwrite=False)

    def test_overwrite_duplicate_allowed(self, registry):
        registry.register("dummy_cam", DummyTestCAMPlugin)
        registry.register("dummy_cam", DummyTestCAMPlugin, overwrite=True)
        assert registry.is_registered("dummy_cam")

    def test_unregister_plugin(self, registry):
        registry.register("dummy_cam", DummyTestCAMPlugin)
        assert registry.unregister("dummy_cam") is True
        assert registry.is_registered("dummy_cam") is False

    def test_unregister_nonexistent_returns_false(self, registry):
        assert registry.unregister("nonexistent_cam") is False

    def test_get_nonexistent_raises(self, registry):
        with pytest.raises(CAMNotFoundError):
            registry.get_plugin_class("nonexistent_cam")


# ── 2. Discovery & Filtering Tests ────────────────────────────────────────────

class TestDiscovery:
    def test_filter_by_modality(self, registry):
        mri_plugins = registry.filter_by_modality("brain_mri")
        assert "gradcam" in mri_plugins
        assert len(mri_plugins) == 7

    def test_filter_by_state(self, registry):
        reg_plugins = registry.filter_by_state(PluginLifecycleState.REGISTERED)
        assert len(reg_plugins) == 7

    def test_filter_by_framework(self, registry):
        pt_plugins = registry.filter_by_framework("pytorch")
        assert len(pt_plugins) == 7

    def test_filter_gradient_free(self, registry):
        gf_plugins = registry.filter_gradient_free()
        assert "scorecam" in gf_plugins
        assert "eigencam" in gf_plugins
        assert "gradcam" not in gf_plugins

    def test_health_check(self, registry):
        health = registry.health_check()
        assert health["total"] == 7
        assert health["registry_healthy"] is True
        assert "gradcam" in health["plugins"]


# ── 3. Lazy Loading Tests ──────────────────────────────────────────────────────

class TestLazyLoading:
    def test_loader_cache_hit(self):
        loader = CAMPluginLoader()
        cls1 = loader.load_class("dummy", DummyTestCAMPlugin)
        cls2 = loader.load_class("dummy", DummyTestCAMPlugin)
        assert cls1 is cls2
        assert loader.is_loaded("dummy")

    def test_loader_unload(self):
        loader = CAMPluginLoader()
        loader.load_class("dummy", DummyTestCAMPlugin)
        assert loader.unload("dummy") is True
        assert not loader.is_loaded("dummy")

    def test_lazy_registration_via_proxy(self, registry):
        registry.load_from_path("proxy_cam", "medical_ai_engine.explainability.plugins.base_cam_plugin", "BaseCAMPlugin")
        assert registry.is_registered("proxy_cam")


# ── 4. Metadata Tests ──────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_to_dict(self):
        meta = CAMPluginMetadata(method_name="test_meta", version="1.2.3")
        d = meta.to_dict()
        assert d["method_name"] == "test_meta"
        assert d["version"] == "1.2.3"
        assert d["status"] == PluginStatus.EXPERIMENTAL.value

    def test_metadata_modality_check(self):
        meta = CAMPluginMetadata(method_name="test", supported_modalities=["brain_mri"])
        assert meta.supports_modality("brain_mri") is True
        assert meta.supports_modality("skin_lesion") is False

    def test_metadata_record_use(self):
        meta = CAMPluginMetadata(method_name="test")
        assert meta.use_count == 0
        meta.record_use()
        assert meta.use_count == 1
        assert meta.last_used_timestamp > 0.0


# ── 5. Lifecycle FSM Tests ────────────────────────────────────────────────────

class TestLifecycleFSM:
    def test_valid_transitions(self, registry):
        registry.transition_state("gradcam", PluginLifecycleState.INITIALIZED)
        assert registry.get_metadata("gradcam").lifecycle_state == PluginLifecycleState.INITIALIZED

        registry.transition_state("gradcam", PluginLifecycleState.READY)
        assert registry.get_metadata("gradcam").lifecycle_state == PluginLifecycleState.READY

        registry.transition_state("gradcam", PluginLifecycleState.ACTIVE)
        assert registry.get_metadata("gradcam").lifecycle_state == PluginLifecycleState.ACTIVE

    def test_invalid_transition_raises(self, registry):
        with pytest.raises(CAMLifecycleError):
            registry.transition_state("gradcam", PluginLifecycleState.ACTIVE)  # REGISTERED -> ACTIVE is invalid


# ── 6. Factory Orchestration Tests ─────────────────────────────────────────────

class TestCAMPluginFactory:
    def test_factory_creates_ready_plugin(self, registry, factory):
        registry.register("dummy", DummyTestCAMPlugin, overwrite=True)
        instance = factory.create("dummy")
        assert isinstance(instance, DummyTestCAMPlugin)
        assert instance.initialize_called is True
        assert instance.ready_called is True
        assert registry.get_metadata("dummy").lifecycle_state == PluginLifecycleState.READY

    def test_factory_activation_deactivation(self, registry, factory):
        registry.register("dummy", DummyTestCAMPlugin, overwrite=True)
        instance = factory.create("dummy")

        factory.activate("dummy", instance)
        assert instance.activate_called is True
        assert registry.get_metadata("dummy").lifecycle_state == PluginLifecycleState.ACTIVE

        factory.deactivate("dummy", instance)
        assert instance.deactivate_called is True
        assert registry.get_metadata("dummy").lifecycle_state == PluginLifecycleState.READY

    def test_factory_destroy(self, registry, factory):
        registry.register("dummy", DummyTestCAMPlugin, overwrite=True)
        instance = factory.create("dummy")

        factory.destroy("dummy", instance)
        assert instance.unload_called is True
        assert registry.get_metadata("dummy").lifecycle_state == PluginLifecycleState.UNLOADED

    def test_factory_reload(self, registry, factory):
        registry.register("dummy", DummyTestCAMPlugin, overwrite=True)
        instance = factory.create("dummy")
        factory.destroy("dummy", instance)

        reloaded_instance = factory.reload("dummy", instance)
        assert reloaded_instance is not instance
        assert registry.get_metadata("dummy").lifecycle_state == PluginLifecycleState.READY


# ── 7. Caching & TTL Tests ─────────────────────────────────────────────────────

class TestCache:
    def test_lru_eviction(self):
        cache = CAMPluginCache(max_capacity=2)
        cache.put("a", "inst_a")
        cache.put("b", "inst_b")
        assert cache.size() == 2

        cache.put("c", "inst_c")  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == "inst_b"
        assert cache.get("c") == "inst_c"

    def test_cache_flush(self):
        cache = CAMPluginCache(max_capacity=5)
        cache.put("a", "inst_a")
        cache.put("b", "inst_b")
        assert cache.flush() == 2
        assert cache.size() == 0

    def test_ttl_expiry(self):
        cache = CAMPluginCache(max_capacity=5, default_ttl=0.01)
        cache.put("a", "inst_a")
        import time
        time.sleep(0.02)
        assert cache.get("a") is None


# ── 8. Thread Safety Tests ────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_registration_and_lookup(self, registry):
        def worker(idx: int):
            key = f"concurrent_cam_{idx}"
            registry.register(key, DummyTestCAMPlugin)
            assert registry.is_registered(key)
            cls = registry.get_plugin_class(key)
            assert cls is DummyTestCAMPlugin

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(registry.list_keys()) == 57  # 7 builtins + 50 workers

    def test_concurrent_factory_instantiation(self, registry, factory):
        registry.register("dummy", DummyTestCAMPlugin, overwrite=True)

        def create_worker():
            return factory.create("dummy", use_cache=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_worker) for _ in range(30)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 30
        # All workers should get the cached singleton instance
        first = results[0]
        for res in results:
            assert res is first


# ── 9. Plugin Validation & Custom Plugin Contract Tests ───────────────────────

class TestPluginValidation:
    def test_custom_plugin_inheriting_base_cam_plugin(self, registry, factory):
        class MyCustomPlugin(BaseCAMPlugin):
            def generate_activation_map(self, model=None, target_layer=None, input_tensor=None, target_category=None, extra_params=None):
                return np.full((224, 224), 0.5, dtype=np.float32)

        registry.register("custom", MyCustomPlugin)
        inst = factory.create("custom")
        arr = inst.generate_activation_map()
        assert arr.shape == (224, 224)
        assert arr[0, 0] == 0.5
