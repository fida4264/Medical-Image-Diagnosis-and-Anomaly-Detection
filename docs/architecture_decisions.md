# Architecture Decision Records — Medical AI Orchestration Engine (Module 14)

> **Project:** Medical Image Diagnosis and Anomaly Detection System  
> **Module:** 14 — Unified Medical AI Orchestration Engine  
> **ADR Count:** 4 (ADR-001 to ADR-004)  
> **Last Updated:** August 1, 2026  

---

## ADR-001: Zero Code Mutation of Modules 1–13

### Status
**ACCEPTED**

### Context
Modules 1–13 are fully implemented, tested (715 passing tests), and production-certified. They represent thousands of lines of validated, debugged code for data engineering, modality classification, and disease-specific AI inference.

### Decision
Module 14 will interact with Modules 1–13 exclusively via **Adapter classes** (`BasePredictorAdapter` subclasses). No existing source file in Modules 1–13 will be modified.

New integrations will use the Adapter pattern — wrapping existing module predictors (`BrainMRIPredictor`, `SkinPredictor`, etc.) and conforming their outputs to the `BasePredictorAdapter` interface.

### Consequences
**Positive:**
- 100% backward compatibility. 715 existing tests remain green without modification.
- Zero regression risk during Module 14 development.
- Each module continues to operate independently via its own CLI (`run_brain_mri.py`, `predict_skin.py`, etc.).
- The orchestration layer is entirely additive — no net reduction in system stability.

**Negative:**
- Each modality requires a thin adapter wrapper (7 adapters × ~30 lines each).
- Adapter output normalization must handle minor schema differences across modules (e.g. Module 11's severity grade, Modules 12/13's reliability score).

---

## ADR-002: Abstract Base Class (`BasePredictorAdapter`) for Engine Uniformity

### Status
**ACCEPTED**

### Context
Modules 7–13 each expose slightly different predictor interfaces and return slightly different JSON schemas. For example:
- Module 11 (`retina`) returns a `severity_grade` field.
- Modules 12 & 13 (`skin`, `ultrasound`) return `prediction_reliability` and `requires_manual_review` fields.
- Module 9 (`brain_ct`) is a feature extraction module without disease labels.

The orchestrator must treat all 7 engines uniformly.

### Decision
Define `BasePredictorAdapter` as an Abstract Base Class with the following contract:

```python
class BasePredictorAdapter(ABC):
    @property
    @abstractmethod
    def modality(self) -> str: ...

    @abstractmethod
    def load_model(self, checkpoint_path: str, label_map_path: str) -> None: ...

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]: ...
```

All 7 modality adapters implement this contract. The orchestrator exclusively interacts with `BasePredictorAdapter`, never directly with module-internal predictor classes.

### Consequences
**Positive:**
- Uniform API — the orchestrator core is decoupled from module-specific implementation details.
- New modalities (e.g. Module 24: Mammography) only need to implement `BasePredictorAdapter`.
- Mock adapters can be used in unit tests without loading actual model weights.

**Negative:**
- Adapter `predict()` return dict must be partially normalized to the unified schema — minor transformation code per adapter.

---

## ADR-003: Soft Fallback for Low Modality Routing Confidence

### Status
**ACCEPTED**

### Context
Module 6 (Modality Classifier) classifies incoming images into 7 modalities. In edge cases (unusual image types, non-standard imaging protocols), the model may produce a low confidence prediction (e.g. 55% `brain_mri`).

Blocking the pipeline on low confidence would degrade user experience and reduce clinical utility.

### Decision
If modality routing confidence is **< 0.70**, the orchestrator will:
1. Continue routing to the top-1 predicted modality.
2. Append a structured warning to `ctx.warnings`: `"Low modality routing confidence (0.55 < threshold 0.70). Clinical review recommended."`
3. Set `ctx.confidence["requires_manual_review"] = True`.
4. Include the warning in the final `UnifiedPredictionResponse`.

Hard blocking (raising an exception and returning `"failed"`) will only occur for **unsupported/unrecognized** modalities (not registered in the `ModalityEngineRegistry`).

### Consequences
**Positive:**
- Prediction still produced even for borderline images — useful in clinical settings where radiologists can manually review.
- `requires_manual_review` flag surfaces the uncertainty to downstream consumers (Gradio UI, REST API clients).

**Negative:**
- Disease prediction for a misclassified modality will be incorrect. Clinical note must clearly communicate this risk.

---

## ADR-004: EngineContext + PipelineStage + EventBus Architecture

### Status
**ACCEPTED**

### Context
The orchestration pipeline requires 12 discrete processing stages:
1. ValidateInput
2. ImageQualityCheck
3. DetectModality
4. RegistryLookup
5. ExecuteInference
6. ConfidenceAnalysis
7. GradCAMHook (optional, for future Module 15)
8. AnomalyHook (optional, for future Module 16)
9. ClinicalReportHook
10. AssemblePredictionJSON
11. Logging
12. Response

Without a formal pattern:
- Each stage would need to pass dozens of parameters to the next stage — tight coupling.
- Future stages (Module 15, 16, 17) would require modifying the orchestrator's method signature each time.
- Lifecycle event hooks (for telemetry, monitoring, Grad-CAM) would need to be hardcoded inside the engine.

### Decision
Adopt three complementary design elements:

#### 1. `EngineContext` (Shared State Carrier)
A single `@dataclass` instance is created per request and passed by reference through all stages. It holds all request state, intermediate outputs, timestamps, logs, errors, performance metrics, and an open `future_extensions` dict.

**No stage needs more than `process(self, ctx: EngineContext) -> None` in its signature.**

#### 2. `BasePipelineStage` (Common Stage Interface)
All stages implement `validate()`, `process()`, `rollback()`, and `cleanup()`. This enables:
- Safe exception handling with guaranteed resource cleanup.
- Future stages (from Module 15+) to be inserted without touching existing stage code.
- Unit testable in isolation with a mock `EngineContext`.

#### 3. `MedicalAIEventBus` (Lifecycle Event Publishing)
The orchestrator publishes named events at key lifecycle points:
`BeforePrediction`, `BeforeRouting`, `AfterRouting`, `BeforeInference`, `AfterInference`, `BeforeReport`, `AfterReport`, `AfterPrediction`, `OnError`, `OnLowConfidence`.

External modules subscribe to these events without modifying Module 14's source code.

### Consequences
**Positive:**
- No parameter sprawl: every stage signature is `(self, ctx: EngineContext)`.
- New stages are additive: insert a new `BasePipelineStage` subclass at any position in the pipeline.
- Module 15 (Grad-CAM) can integrate by: (a) adding `GradCAMHookStage` to the pipeline + (b) subscribing to `AfterInference` via `EventBus`. Module 14's core source is untouched.
- Full audit trail per request via `ctx.logs`, `ctx.errors`, `ctx.timestamps`, and `ctx.performance_metrics`.
- `ctx.future_extensions` absorbs new data fields from Modules 15–19 without schema breakage.

**Negative:**
- `EngineContext` is mutable — poorly written stages could corrupt shared state. Mitigation: comprehensive unit tests per stage with mock context injection.
- Event bus is synchronous (in-process). If a subscriber handler is slow, it blocks the pipeline. Mitigation: define strict handler time budgets (< 1 ms) or offload to async queue for non-critical subscribers (Module 19 database logging).

---

*Architecture Decision Records — Medical AI Orchestration Engine (Module 14)*  
*Version 1.1 — 4 ADRs Documented — August 1, 2026*
