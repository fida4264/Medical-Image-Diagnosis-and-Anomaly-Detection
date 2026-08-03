"""
Medical AI Engine Registry — Test Runner Script (Module 14.2.2)

Demonstrates:
  1. ModelRegistry creation and adapter registration
  2. Lazy loading via get_adapter()
  3. Cache hit on second get_adapter()
  4. Cache invalidation and reload
  5. Health check across all registered adapters
  6. Metadata retrieval
  7. PyTest suite execution
"""

import sys
import pytest
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration


def _make_mock_adapter(modality_key: str):
    """Create a zero-dependency mock adapter for demo."""
    from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter
    from typing import Any, Dict, Optional
    from PIL import Image

    class _MockAdapter(BasePredictorAdapter):
        _MODALITY = modality_key

        @property
        def modality(self):
            return self._MODALITY

        def load_model(self, checkpoint_path=None, label_map_path=None):
            self._is_loaded = True
            self._predictor = object()

        def predict(self, image):
            return {"modality": self.modality, "predicted_class": "demo_class", "confidence": 0.99}

        def get_metadata(self):
            m = super().get_metadata()
            m["supported_classes"] = ["class_a", "class_b"]
            return m

    _MockAdapter.__name__ = f"Mock{modality_key.title().replace('_','')}Adapter"
    return _MockAdapter


def main():
    print("=" * 66)
    print("   Module 14.2.2: Model Registry & Adapter Framework Runner   ")
    print("=" * 66)

    ModelRegistry.reset_instance()
    cfg = RegistryConfiguration(auto_discover_adapters=False)
    reg = ModelRegistry(config=cfg)

    MODALITIES = ["brain_mri", "chest_xray", "brain_ct", "histopathology", "retina", "skin", "ultrasound"]

    print("\n[1/6] Registering 7 mock adapters...")
    for mod in MODALITIES:
        reg.register_adapter(mod, _make_mock_adapter(mod))
    print(f"      Registered: {reg.list_registered_modalities()}")

    print("\n[2/6] Lazy-loading skin adapter (first access)...")
    skin_adapter = reg.get_adapter("skin")
    print(f"      Adapter loaded: {skin_adapter.is_loaded}")
    assert skin_adapter.is_loaded

    print("\n[3/6] Cache hit on second access (same instance)...")
    skin_adapter_2 = reg.get_adapter("skin")
    assert skin_adapter is skin_adapter_2
    print(f"      Same instance: {skin_adapter is skin_adapter_2}")

    print("\n[4/6] Cache invalidation + reload for retina...")
    reg.invalidate_cache("retina")
    retina_adapter = reg.reload_adapter("retina")
    print(f"      Retina reloaded: {retina_adapter.is_loaded}")
    assert retina_adapter.is_loaded

    print("\n[5/6] Health check across registry...")
    health = reg.health_check()
    print(f"      Overall status:   {health['overall_status']}")
    print(f"      Registered count: {health['registered_count']}")
    print(f"      Loaded count:     {health['loaded_count']}")

    print("\n[6/6] Metadata retrieval for brain_mri...")
    meta = reg.get_adapter_metadata("brain_mri")
    print(f"      Modality: {meta['modality']}")
    print(f"      Status:   {meta['status']}")

    print("\n" + "-" * 66)
    print("Running PyTest Suite...")
    print("-" * 66)

    ModelRegistry.reset_instance()

    ret = pytest.main([
        "medical_ai_engine/tests/test_registry.py",
        "-v",
        "--tb=short",
    ])

    if ret == 0:
        print("\n[SUCCESS] All Module 14.2.2 Registry & Adapter tests PASSED!")
    else:
        print(f"\n[FAILED] PyTest exit code: {ret}")
        sys.exit(ret)


if __name__ == "__main__":
    main()
