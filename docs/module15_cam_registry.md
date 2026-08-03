# Module 15.2 — CAM Registry & Plugin System

**Medical AI Diagnosis and Anomaly Detection Platform**
**Phase 4 — Explainable AI (XAI)**
**Status:** ✅ COMPLETE · 82/82 Tests Passed (30 Module 15.2 + 52 Module 15.1)

---

## 1. Executive Summary

Module 15.2 delivers the production **CAM Registry & Plugin Infrastructure** for the Medical AI Platform. It provides a thread-safe, decoupled plugin environment for discovering, loading, instantiating, caching, and managing Explainability (CAM) methods without directly importing any concrete algorithm implementation.

### Key Capabilities
- **Dynamic Registration & Unregistration:** Thread-safe runtime registration with duplicate protection.
- **Lazy Loading:** `CAMPluginLoader` defers Python module import until `get_plugin_class()` is called.
- **Plugin Discovery & Filtering:** Query by modality, framework (PyTorch), state, or gradient-free status.
- **Metadata Management:** Comprehensive metadata tracking via `CAMPluginMetadata`.
- **Lifecycle FSM:** Formal finite state machine governing transitions from `UNREGISTERED` to `RELOADED`.
- **Factory Orchestration:** `CAMPluginFactory` handles construction, config injection, and lifecycle transitions.
- **Thread-Safe LRU Cache:** `CAMPluginCache` with TTL expiry, LRU eviction, and hit/miss statistics.
- **Hot-Swap Support:** Prepared infrastructure for future live reloading without process restarts.

---

## 2. Architecture & Package Structure

```
medical_ai_engine/
└── explainability/
    ├── interfaces/
    │   ├── cam_registry_interface.py   ← ICAMRegistry (ABC)
    │   └── cam_plugin_interface.py     ← ICAMPlugin (ABC)
    ├── plugins/
    │   ├── __init__.py
    │   └── base_cam_plugin.py          ← BaseCAMPlugin (Concrete Base)
    ├── registry/
    │   ├── __init__.py                 ← Package exports
    │   ├── cam_registry.py             ← CAMProductionRegistry (Singleton)
    │   ├── cam_factory.py              ← CAMPluginFactory
    │   ├── cam_loader.py               ← CAMPluginLoader
    │   ├── cam_cache.py                ← CAMPluginCache
    │   ├── cam_metadata.py              ← CAMPluginMetadata & FSM
    │   ├── cam_configuration.py         ← CAMPluginConfiguration
    │   └── cam_exceptions.py           ← Exception Hierarchy
    └── tests/
        └── test_cam_registry.py        ← 30 Unit Tests
```

---

## 3. Plugin Lifecycle State Machine

Plugins progress through a formal FSM enforced by `CAMProductionRegistry` and `CAMPluginFactory`:

```
UNREGISTERED ────► REGISTERED ────► INITIALIZED ────► READY ────► ACTIVE
                       │                │               │          │
                       ▼                ▼               ▼          ▼
                    UNLOADED ◄─────────────────────────────────────┘
                       │
                       ▼
                    RELOADED ───► READY
```

### State Definitions

| State | Description | Allowed Next States |
|---|---|---|
| `UNREGISTERED` | Initial state prior to registration | `REGISTERED` |
| `REGISTERED` | Class & metadata registered; not initialized | `INITIALIZED`, `UNLOADED` |
| `INITIALIZED` | Model hooks/buffers allocated | `READY`, `UNLOADED` |
| `READY` | Configured and warmed up for requests | `ACTIVE`, `INITIALIZED`, `UNLOADED` |
| `ACTIVE` | Currently processing an active request | `READY`, `UNLOADED` |
| `UNLOADED` | Explicitly torn down; resources freed | `RELOADED`, `REGISTERED` |
| `RELOADED` | Hot-swapped; returning to service | `READY`, `INITIALIZED` |

---

## 4. Component Details

### 4.1 `CAMProductionRegistry`

The primary singleton registry satisfying `ICAMRegistry`. It tracks class references and metadata dicts with thread-safe `RLock` guards.

```python
from medical_ai_engine.explainability.registry import CAMProductionRegistry

registry = CAMProductionRegistry.get_instance()

# Check built-in registered stubs
print(registry.list_keys())
# ['eigencam', 'gradcam', 'gradcam_plus_plus', 'guided_backprop', 'integrated_gradients', 'layercam', 'scorecam']

# Filter plugins by modality
mri_plugins = registry.filter_by_modality("brain_mri")

# Filter gradient-free plugins
grad_free = registry.filter_gradient_free() # ['eigencam', 'scorecam']
```

### 4.2 `CAMPluginMetadata`

Tracks identity, capability flags, and runtime statistics:

| Field | Type | Description |
|---|---|---|
| `method_name` | `str` | Unique key (e.g. `'gradcam'`) |
| `version` | `str` | Semantic version string |
| `requires_gradients` | `bool` | True if autograd required |
| `gpu_support` | `bool` | GPU capability flag |
| `supported_modalities` | `List[str]` | List of supported modality keys |
| `lifecycle_state` | `PluginLifecycleState` | Current FSM state |
| `use_count` | `int` | Total invocation count |
| `status` | `PluginStatus` | Deployment status (`STABLE`, `BETA`, etc.) |

### 4.3 `CAMPluginFactory`

Instantiates plugins, injects per-method `CAMPluginConfiguration`, and drives the FSM:

```python
from medical_ai_engine.explainability.registry import CAMPluginFactory, CAMProductionRegistry

registry = CAMProductionRegistry.get_instance()
factory = CAMPluginFactory(registry=registry)

# Request a plugin (factory checks LRU cache first)
plugin = factory.create("gradcam")  # Transitions to READY

# Lifecycle callbacks
factory.activate("gradcam", plugin)    # READY -> ACTIVE
# ... run prediction explanation ...
factory.deactivate("gradcam", plugin)  # ACTIVE -> READY
```

### 4.4 `CAMPluginCache`

Thread-safe LRU cache storing plugin instances to avoid re-instantiation overhead.

- **Capacity:** Default 7 entries (configurable).
- **Eviction Policy:** Least-Recently-Used (LRU).
- **TTL Expiry:** Optional per-entry time-to-live.

### 4.5 `CAMPluginLoader`

Lazy loader supporting dynamic imports from module paths:

```python
registry.load_from_path(
    plugin_key="gradcam",
    module_path="medical_ai_engine.explainability.methods.gradcam",
    class_name="GradCAMMethod",
)
```

---

## 5. Supported Plugin Types (Infrastructure Prepared)

The registry pre-registers metadata and stubs for all 7 supported CAM methods:

1. **GradCAM** (`gradcam`) — Gradient-weighted Class Activation Mapping
2. **GradCAM++** (`gradcam_plus_plus`) — Improved localization for multi-instance targets
3. **LayerCAM** (`layercam`) — Fine-grained multi-layer activation maps
4. **ScoreCAM** (`scorecam`) — Gradient-free activation-based maps
5. **EigenCAM** (`eigencam`) — Principal component activation maps
6. **Integrated Gradients** (`integrated_gradients`) — Axiomatic attribution method
7. **Guided Backpropagation** (`guided_backprop`) — High-resolution saliency maps

> ℹ️ *Note: Algorithms will be implemented in Module 15.3+.*

---

## 6. Exception Hierarchy

```
CAMException
├── CAMRegistrationError
│   ├── CAMDuplicateError
│   └── CAMInvalidPluginError
├── CAMLoadError
│   └── CAMModuleNotFoundError
├── CAMFactoryError
│   └── CAMLifecycleError
├── CAMCacheError
├── CAMValidationError
└── CAMNotFoundError
```

---

## 7. Verification & Test Results

```
============================= test session starts =============================
medical_ai_engine/explainability/tests/test_cam_registry.py PASSED (30 tests)
medical_ai_engine/explainability/tests/test_explainability_architecture.py PASSED (52 tests)
============================= 82 passed in 0.57s ==============================
```

- **Registration & Duplicate Detection:** 7 tests PASSED
- **Discovery & Filtering:** 5 tests PASSED
- **Lazy Loading & Loader:** 3 tests PASSED
- **Metadata:** 3 tests PASSED
- **Lifecycle FSM:** 2 tests PASSED
- **Factory Orchestration:** 4 tests PASSED
- **Caching & TTL:** 3 tests PASSED
- **Thread Safety:** 2 tests PASSED (concurrent registration & creation under 10 workers)
- **Plugin Validation:** 1 test PASSED
