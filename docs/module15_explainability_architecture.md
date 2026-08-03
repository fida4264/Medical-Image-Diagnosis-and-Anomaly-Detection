# Module 15.1 — Explainability Framework Architecture

**Medical AI Diagnosis and Anomaly Detection Platform**
**Phase 4 — Explainable AI (XAI)**
**Status:** ✅ COMPLETE · 52/52 Tests Passed

---

## 1. Overview

Module 15.1 introduces a **production-grade, plugin-based Explainable AI (XAI) framework** that integrates seamlessly with the existing `MedicalAIEngine` without modifying any prior module.

The framework provides:
- A 6-stage orchestration pipeline that transforms a raw prediction into a clinically-grounded visual explanation.
- A singleton `CAMRegistry` / `CAMFactory` enabling zero-configuration plugin registration of any future CAM method.
- Decoupled event-bus integration via `ExplainabilityEventListener` subscribing to `EngineCompleted` events.
- Immutable output (`ExplainabilityResult`) delivered as both Python dataclass and JSON.

> **Backward Compatibility:** Zero modifications to Modules 1–14. No existing code was changed.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       MedicalAIEngine                           │
│        (Module 14 — unmodified, publishes EngineCompleted)      │
└────────────────────────┬────────────────────────────────────────┘
                         │  EngineCompleted event
                         ▼
          ┌──────────────────────────────┐
          │  ExplainabilityEventListener │  ← async thread
          └──────────────┬───────────────┘
                         │ builds ExplainabilityContext
                         ▼
          ┌──────────────────────────────┐
          │     ExplainabilityEngine     │  ← public API
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │   ExplainabilityPipeline     │
          │                              │
          │  Stage 1: Validation         │
          │  Stage 2: Layer Resolution   │  ← TargetLayerManager
          │  Stage 3: Activation Extract │  ← CAMFactory → BaseCAMMethod
          │  Stage 4: Heatmap Generation │  ← HeatmapGenerator
          │  Stage 5: Overlay Generation │  ← OverlayGenerator
          │  Stage 6: Clinical Report    │  ← ClinicalExplanationBuilder
          └──────────────┬───────────────┘
                         │
                         ▼
            ExplainabilityResult (frozen dataclass)
                         │
                         ▼
            ExplainabilityJSONBuilder → JSON response
```

---

## 3. Package Structure

```
medical_ai_engine/
└── explainability/
    ├── __init__.py                       ← Public API exports
    ├── interfaces/
    │   ├── __init__.py
    │   ├── cam_method_interface.py       ← ICAMMethod (ABC)
    │   └── explainability_engine_interface.py ← IExplainabilityEngine (ABC)
    ├── explainability_configuration.py   ← Config dataclass
    ├── explainability_context.py         ← Mutable shared state
    ├── explainability_result.py          ← Immutable frozen output
    ├── base_cam_method.py                ← BaseCAMMethod + 7 stubs
    ├── cam_registry.py                   ← Singleton plugin registry
    ├── cam_factory.py                    ← Instance factory
    ├── target_layer_manager.py           ← Layer resolution logic
    ├── heatmap_generator.py              ← Activation → RGB heatmap
    ├── overlay_generator.py              ← Alpha-blend overlay
    ├── clinical_explanation_builder.py   ← Anatomical region mapping
    ├── explainability_json_builder.py    ← JSON/dict serialization
    ├── explainability_pipeline.py        ← 6-stage pipeline orchestrator
    ├── explainability_engine.py          ← Primary engine facade
    ├── explainability_event_listener.py  ← Event bus subscriber
    └── tests/
        ├── __init__.py
        └── test_explainability_architecture.py  ← 52 tests
```

---

## 4. Key Components

### 4.1 Interfaces

| Interface | Role |
|---|---|
| `ICAMMethod` | Contract for all CAM algorithm implementations |
| `IExplainabilityEngine` | Contract for the engine facade |

```python
class ICAMMethod(ABC):
    @abstractmethod
    def generate_activation_map(self, model, target_layer, input_tensor) -> np.ndarray: ...

    @abstractmethod
    def get_metadata(self) -> dict: ...
```

### 4.2 ExplainabilityConfiguration

`@dataclass` holding all tunable parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `default_cam_method` | `"gradcam"` | Method used when none specified |
| `supported_methods` | 7 methods | Registered CAM methods |
| `default_target_layers` | 7 modality mappings | Per-modality layer selection |
| `colormap` | `"jet"` | Heatmap colormap |
| `overlay_alpha` | `0.4` | Blending coefficient |
| `min_confidence_for_explanation` | `0.5` | Skip threshold |
| `enabled` | `True` | Master on/off switch |

### 4.3 ExplainabilityContext

Mutable shared state object threaded through all pipeline stages:

```python
ctx = ExplainabilityContext(
    request_id="xai-001",
    session_id="sess-001",
    image=pil_image,
    modality="brain_mri",
    predicted_class="tumor",
    confidence=0.92,
    cam_method_key="gradcam",
)
```

### 4.4 ExplainabilityResult

`@dataclass(frozen=True)` — immutable output:

```python
result = engine.explain(ctx)
print(result.status)              # "SUCCESS"
print(result.clinical_regions)   # list of anatomical region dicts
print(result.clinical_summary)   # human-readable explanation
print(result.to_json())          # serialized JSON
```

### 4.5 Plugin Registry

```python
# Register a custom CAM method at runtime
from medical_ai_engine.explainability import CAMRegistry, BaseCAMMethod

class MyNewCAM(BaseCAMMethod):
    def __init__(self):
        super().__init__(name="my_cam", requires_gradients=False)

CAMRegistry.get_instance().register("my_cam", MyNewCAM)
```

### 4.6 Event-Bus Integration

```python
listener = ExplainabilityEventListener(
    config=ExplainabilityConfiguration.default(),
    result_callback=lambda result: store(result),
    async_execution=True,
)
listener.subscribe(engine_integrator.engine.event_publisher)
```

---

## 5. Modality → Target Layer Mapping

| Modality | Default Target Layer |
|---|---|
| `brain_mri` | `layer4` |
| `chest_xray` | `features` |
| `skin_lesion` | `features.denseblock4` |
| `eye_retinal` | `mixed10` |
| `breast_mri` | `layer4` |
| `ct_scan` | `layer4` |
| `pathology` | `layer4` |

---

## 6. Supported CAM Methods (Stubs — Implementations in Module 15.2)

| Key | Class | Requires Gradients |
|---|---|---|
| `gradcam` | `GradCAMMethodStub` | ✅ Yes |
| `gradcam_plusplus` | `GradCAMPlusPlusMethodStub` | ✅ Yes |
| `layercam` | `LayerCAMMethodStub` | ✅ Yes |
| `scorecam` | `ScoreCAMMethodStub` | ❌ No |
| `eigencam` | `EigenCAMMethodStub` | ❌ No |
| `integrated_gradients` | `IntegratedGradientsMethodStub` | ✅ Yes |
| `guided_backprop` | `GuidedBackpropMethodStub` | ✅ Yes |

---

## 7. Pipeline Execution Flow

```
ExplainabilityContext (CREATED)
    │
    ▼ Stage 1 — Validation
    │  Checks: enabled, confidence >= threshold, modality set
    │
    ▼ Stage 2 — Layer Resolution
    │  TargetLayerManager resolves: explicit override → config map → default
    │
    ▼ Stage 3 — Activation Extraction
    │  CAMFactory creates method from cam_method_key
    │  BaseCAMMethod.generate_activation_map() returns 2D float32 [0,1]
    │
    ▼ Stage 4 — Heatmap Generation
    │  HeatmapGenerator applies colormap → (H,W,3) RGB array + PIL Image
    │
    ▼ Stage 5 — Overlay Generation
    │  OverlayGenerator alpha-blends heatmap onto original image
    │
    ▼ Stage 6 — Clinical Report
    │  ClinicalExplanationBuilder maps activations to anatomical regions
    │  Returns: List[RegionDict] + clinical_summary string
    │
    ▼
ExplainabilityResult (SUCCESS / FAILED / SKIPPED)
```

---

## 8. Test Results

```
============================= test session starts =============================
collected 52 items

TestExplainabilityConfiguration      7 passed
TestExplainabilityContext             5 passed
TestExplainabilityResult              5 passed
TestBaseCAMMethod                     4 passed
TestCAMRegistry                       8 passed
TestCAMFactory                        4 passed
TestTargetLayerManager                3 passed
TestHeatmapGenerator                  2 passed
TestOverlayGenerator                  2 passed
TestClinicalExplanationBuilder        2 passed
TestExplainabilityJSONBuilder         2 passed
TestExplainabilityEngine              5 passed
TestPackageImports                    3 passed
──────────────────────────────────────────────────────
52 passed in 0.34s                    ✅ 100% PASS RATE
```

---

## 9. Design Principles

| Principle | Implementation |
|---|---|
| **Plugin-Based** | `CAMRegistry` singleton accepts runtime `register()` calls |
| **Decoupled** | Connects to MedicalAIEngine via event bus only (no import) |
| **Immutable Output** | `ExplainabilityResult` is `frozen=True` |
| **Async-Safe** | `ExplainabilityEventListener` launches daemon threads |
| **Zero Modification** | Modules 1–14 unchanged; backward-compatible |
| **Extensible** | All stubs swapped by real implementations in Module 15.2 |

---

## 10. Next Steps

**Module 15.2 — Grad-CAM & CAM Implementation Suite** will:
- Replace all 7 stubs with production PyTorch hook-based implementations
- Integrate with `torch.nn.Module` hook system for real gradient capture
- Validate against clinical benchmark datasets

> ⚠️ **Awaiting explicit user approval before proceeding to Module 15.2.**
