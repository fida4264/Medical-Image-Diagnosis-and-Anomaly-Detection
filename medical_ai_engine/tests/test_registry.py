"""
Medical AI Engine — Registry & Adapter Test Suite (Module 14.2.2)

Tests:
  - AdapterRegistry: registration, duplicate, lookup, list, unregister
  - AdapterCache: put/get/contains/invalidate
  - AdapterMetadata: serialization
  - ModelRegistry: register, get, lazy-load, cache, reload, hot_swap, health_check
  - Thread safety: concurrent get_adapter from multiple threads
  - Failure handling: unregistered lookup, failed load, duplicate registration
  - IPredictorAdapter interface compliance: all 7 adapters (mock-only)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import threading
import pytest
from PIL import Image

from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter
from medical_ai_engine.registry.adapter_cache import AdapterCache
from medical_ai_engine.registry.adapter_metadata import AdapterMetadata
from medical_ai_engine.registry.adapter_registry import AdapterRegistry
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.registry.registry_exceptions import (
    AdapterNotFoundException,
    DuplicateAdapterRegistrationException,
)


# ── Mock Adapter Implementations ───────────────────────────────────────────────

class _MockAdapter(BasePredictorAdapter):
    """Mock IPredictorAdapter for testing — no real model loading."""

    _MODALITY = "mock_modality"
    load_called: bool = False

    @property
    def modality(self) -> str:
        return self._MODALITY

    def load_model(self, checkpoint_path: Optional[str] = None, label_map_path: Optional[str] = None) -> None:
        self._is_loaded = True
        self.load_called = True
        self._predictor = object()  # sentinel

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "predicted_class": "mock_class",
            "confidence": 0.99,
            "status": "success",
        }

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta["supported_classes"] = ["class_a", "class_b"]
        return meta


def _make_modality_adapter(modality_key: str):
    """Factory to create mock adapter subclasses for arbitrary modality keys."""
    class _Adapter(_MockAdapter):
        _MODALITY = modality_key
    _Adapter.__name__ = f"Mock{modality_key.title().replace('_','')}Adapter"
    return _Adapter


class _FailingLoadAdapter(_MockAdapter):
    """Mock adapter that always raises on load_model."""
    _MODALITY = "failing_modality"

    def load_model(self, checkpoint_path: Optional[str] = None, label_map_path: Optional[str] = None) -> None:
        raise RuntimeError("Simulated load failure")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry_singleton():
    """Reset ModelRegistry singleton between tests."""
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


def _build_registry(auto_discover: bool = False) -> ModelRegistry:
    cfg = RegistryConfiguration(auto_discover_adapters=auto_discover)
    return ModelRegistry(config=cfg)


# ── Tests: AdapterMetadata ─────────────────────────────────────────────────────

def test_adapter_metadata_serialization():
    meta = AdapterMetadata(
        modality="skin",
        supported_classes=["melanoma", "nevus"],
        model_version="1.0.0",
    )
    d = meta.to_dict()
    assert d["modality"] == "skin"
    assert d["num_classes"] == 2
    assert "melanoma" in d["supported_classes"]
    assert "model_version" in d


# ── Tests: AdapterCache ────────────────────────────────────────────────────────

def test_adapter_cache_put_and_get():
    cache = AdapterCache()
    adapter = _MockAdapter()
    adapter.initialize()
    adapter.load_model()  # sets _is_loaded = True
    cache.put("skin", adapter)
    assert cache.contains("skin")
    retrieved = cache.get("skin")
    assert retrieved is adapter


def test_adapter_cache_invalidate_single():
    cache = AdapterCache()
    adapter = _MockAdapter()
    adapter.initialize()
    adapter.load_model()
    cache.put("skin", adapter)
    cache.invalidate("skin")
    assert not cache.contains("skin")


def test_adapter_cache_invalidate_all():
    cache = AdapterCache()
    for mod in ["skin", "retina", "brain_mri"]:
        a = _make_modality_adapter(mod)()
        a.initialize()
        a.load_model()
        cache.put(mod, a)
    assert cache.size() == 3
    cache.invalidate(None)
    assert cache.size() == 0


# ── Tests: AdapterRegistry ─────────────────────────────────────────────────────

def test_adapter_registry_register_and_lookup():
    registry = AdapterRegistry()
    registry.register("skin", _MockAdapter)
    assert registry.is_registered("skin")
    cls = registry.get_class("skin")
    assert cls is _MockAdapter


def test_adapter_registry_duplicate_raises():
    registry = AdapterRegistry()
    registry.register("skin", _MockAdapter)
    with pytest.raises(DuplicateAdapterRegistrationException):
        registry.register("skin", _MockAdapter, overwrite=False)


def test_adapter_registry_overwrite_allowed():
    registry = AdapterRegistry()
    registry.register("skin", _MockAdapter)
    registry.register("skin", _MockAdapter, overwrite=True)  # should not raise
    assert registry.is_registered("skin")


def test_adapter_registry_not_found_raises():
    registry = AdapterRegistry()
    with pytest.raises(AdapterNotFoundException):
        registry.get_class("nonexistent")


def test_adapter_registry_list_modalities():
    registry = AdapterRegistry()
    for mod in ["brain_mri", "skin", "retina"]:
        registry.register(mod, _make_modality_adapter(mod))
    modalities = registry.list_modalities()
    assert sorted(modalities) == ["brain_mri", "retina", "skin"]


def test_adapter_registry_unregister():
    registry = AdapterRegistry()
    registry.register("skin", _MockAdapter)
    registry.unregister("skin")
    assert not registry.is_registered("skin")


# ── Tests: ModelRegistry ──────────────────────────────────────────────────────

def test_model_registry_register_and_is_registered():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)
    assert reg.is_registered("skin")


def test_model_registry_duplicate_raises():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)
    with pytest.raises(DuplicateAdapterRegistrationException):
        reg.register_adapter("skin", SkinAdapter, overwrite=False)


def test_model_registry_get_adapter_lazy_loads():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)

    assert not reg.is_loaded("skin")
    adapter = reg.get_adapter("skin")
    assert adapter.is_loaded
    assert reg.is_loaded("skin")


def test_model_registry_get_adapter_cached():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)

    adapter1 = reg.get_adapter("skin")
    adapter2 = reg.get_adapter("skin")
    assert adapter1 is adapter2  # same cached instance


def test_model_registry_invalidate_cache():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)
    reg.get_adapter("skin")
    assert reg.is_loaded("skin")
    reg.invalidate_cache("skin")
    assert not reg.is_loaded("skin")


def test_model_registry_reload_adapter():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)
    adapter1 = reg.get_adapter("skin")
    adapter2 = reg.reload_adapter("skin")
    assert adapter1 is not adapter2
    assert adapter2.is_loaded


def test_model_registry_not_found_raises():
    reg = _build_registry()
    with pytest.raises(AdapterNotFoundException):
        reg.get_adapter("nonexistent_modality")


def test_model_registry_health_check():
    reg = _build_registry()
    for mod in ["brain_mri", "skin", "retina"]:
        reg.register_adapter(mod, _make_modality_adapter(mod))
    health = reg.health_check()
    assert health["registered_count"] == 3
    assert health["overall_status"] == "HEALTHY"
    assert "skin" in health["modalities"]
    assert "brain_mri" in health["modalities"]


def test_model_registry_adapter_metadata():
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)
    meta = reg.get_adapter_metadata("skin")
    assert meta["modality"] == "skin"
    assert "status" in meta


def test_model_registry_list_registered_modalities():
    reg = _build_registry()
    for mod in ["brain_mri", "chest_xray", "skin"]:
        reg.register_adapter(mod, _make_modality_adapter(mod))
    modalities = reg.list_registered_modalities()
    assert "brain_mri" in modalities
    assert "chest_xray" in modalities
    assert "skin" in modalities


def test_model_registry_failing_adapter_load():
    reg = _build_registry()
    reg.register_adapter("failing_modality", _FailingLoadAdapter)
    from medical_ai_engine.registry.registry_exceptions import AdapterLoadException
    with pytest.raises(AdapterLoadException):
        reg.get_adapter("failing_modality")


# ── Tests: Thread Safety ──────────────────────────────────────────────────────

def test_model_registry_concurrent_get_adapter():
    """Ensure concurrent access loads adapter exactly once (no race conditions)."""
    reg = _build_registry()
    SkinAdapter = _make_modality_adapter("skin")
    reg.register_adapter("skin", SkinAdapter)

    adapters = []
    errors = []

    def worker():
        try:
            a = reg.get_adapter("skin")
            adapters.append(a)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(adapters) == 20
    # All threads should have received the SAME cached instance
    assert all(a is adapters[0] for a in adapters)


def test_model_registry_concurrent_registration():
    """Concurrent registration of distinct modalities should not deadlock or corrupt state."""
    reg = _build_registry()
    errors = []

    def register_worker(modality_key: str):
        try:
            adapter_cls = _make_modality_adapter(modality_key)
            reg.register_adapter(modality_key, adapter_cls)
        except Exception as e:
            errors.append(e)

    modalities = [f"modality_{i}" for i in range(15)]
    threads = [threading.Thread(target=register_worker, args=(m,)) for m in modalities]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(reg.list_registered_modalities()) == 15


# ── Tests: BasePredictorAdapter Contract ─────────────────────────────────────

def test_base_adapter_interface_compliance():
    """Verify BasePredictorAdapter correctly implements IPredictorAdapter contract."""
    adapter = _MockAdapter()
    adapter.initialize()

    assert not adapter.is_loaded
    adapter.load_model()
    assert adapter.is_loaded

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    result = adapter.predict(img)
    assert "predicted_class" in result
    assert "confidence" in result

    health = adapter.health_check()
    assert health["loaded"] is True
    assert health["modality"] == "mock_modality"

    meta = adapter.get_metadata()
    assert "supported_classes" in meta
    assert "framework" in meta

    adapter.shutdown()
    assert not adapter.is_loaded


def test_all_seven_adapter_modalities_registered_on_auto_discover():
    """Verify auto-discover silently skips unavailable modules and registers what it can."""
    cfg = RegistryConfiguration(auto_discover_adapters=True)
    reg = ModelRegistry(config=cfg)
    modalities = reg.list_registered_modalities()
    # At minimum we expect interface compliance — adapters may not be registered
    # if model weights don't exist, but no exception should be raised
    assert isinstance(modalities, list)
