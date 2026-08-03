# Design Patterns — Medical AI Orchestration Engine (Module 14)

> **Module:** 14 — Unified Medical AI Orchestration Engine  
> **Specification Version:** 1.1  
> **Last Updated:** August 1, 2026  

---

## Overview

Module 14 employs **8 design patterns** to achieve zero-coupling extensibility, uniform interfaces, and production-grade reliability without modifying any of the 13 existing modules.

---

## 1. Registry Pattern — `ModalityEngineRegistry`

### Intent
Maintain a central, runtime-discoverable catalogue of all registered modality predictor adapters.

### Structure
```python
class ModalityEngineRegistry:
    _registry: Dict[str, Type[BasePredictorAdapter]] = {}   # class registrations
    _instances: Dict[str, BasePredictorAdapter] = {}        # cached loaded instances
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, modality: str, adapter_cls: Type[BasePredictorAdapter]) -> None:
        cls._registry[modality.lower()] = adapter_cls

    @classmethod
    def get(cls, modality: str) -> BasePredictorAdapter:
        """Returns cached instance, loading lazily on first access."""
        with cls._lock:
            if modality not in cls._instances:
                cls._instances[modality] = cls._registry[modality]()
                cls._instances[modality].load_model(...)
            return cls._instances[modality]

    @classmethod
    def reload(cls, modality: str) -> None:
        with cls._lock:
            del cls._instances[modality]

    @classmethod
    def swap(cls, modality: str, new_checkpoint: str) -> None:
        """Zero-downtime model hot-swap."""
        new_instance = cls._registry[modality]()
        new_instance.load_model(new_checkpoint, ...)
        with cls._lock:
            cls._instances[modality] = new_instance
```

### Why It's Used
- New modality engines (e.g. Module 14.x: Mammography) register themselves without modifying the orchestrator.
- Thread-safe lazy loading prevents VRAM exhaustion from loading all 7 models at startup.

---

## 2. Factory Pattern — `PredictorAdapterFactory`

### Intent
Encapsulate object creation logic for adapters, decoupling the orchestrator from concrete adapter class names.

### Structure
```python
class PredictorAdapterFactory:
    @staticmethod
    def create(modality: str, config: Optional[dict] = None) -> BasePredictorAdapter:
        adapter_cls = ModalityEngineRegistry.get_class(modality)
        if adapter_cls is None:
            raise UnsupportedModalityError(modality)
        return adapter_cls.from_config(config)
```

### Why It's Used
- The orchestrator calls `PredictorAdapterFactory.create("skin")` without knowing which concrete class it's instantiating.
- Configuration (checkpoint path, backend) is injected at factory time.

---

## 3. Strategy Pattern — `InferenceStrategy`

### Intent
Define a family of interchangeable inference backends (PyTorch, ONNX, TorchScript) and make them swappable at configuration time without changing the adapter code.

### Structure
```python
class BaseInferenceStrategy(ABC):
    @abstractmethod
    def run(self, model_path: str, input_tensor: Tensor) -> Tensor: ...

class PyTorchInferenceStrategy(BaseInferenceStrategy):
    def run(self, model_path, input_tensor): ...  # loads .pt, runs forward()

class ONNXInferenceStrategy(BaseInferenceStrategy):
    def run(self, model_path, input_tensor): ...  # loads .onnx, runs ort.InferenceSession

class TorchScriptInferenceStrategy(BaseInferenceStrategy):
    def run(self, model_path, input_tensor): ...  # loads .torchscript, runs traced model
```

### Configuration
```yaml
# configs/orchestration.yaml
orchestrator:
  default_backend: "pytorch"  # pytorch | onnx | torchscript
```

### Why It's Used
- Allows switching to ONNX runtime for CPU-only deployment environments without code changes.
- Each adapter selects its backend via `from_config()` — strategy is injected, not hardcoded.

---

## 4. Pipeline Pattern — `OrchestrationPipeline` + `BasePipelineStage`

### Intent
Define a sequential chain of processing stages where each stage receives and mutates a shared `EngineContext` object.

### Structure
```python
class OrchestrationPipeline:
    def __init__(self, stages: List[BasePipelineStage]):
        self.stages = stages

    def run(self, ctx: EngineContext) -> None:
        for stage in self.stages:
            try:
                stage.validate(ctx)
                stage.process(ctx)
            except Exception as e:
                stage.rollback(ctx)
                raise
            finally:
                stage.cleanup(ctx)
```

### Default Stage Order (Module 14.2)
```
ValidateInputStage
→ ImageQualityCheckStage
→ DetectModalityStage
→ RegistryLookupStage
→ ExecuteInferenceStage
→ ConfidenceAnalysisStage
→ GradCAMHookStage         (optional — enabled by config)
→ AnomalyHookStage          (optional — enabled by config)
→ ClinicalReportHookStage
→ AssemblePredictionJSONStage
→ LoggingStage
→ ResponseStage
```

### Why It's Used
- Stages are completely decoupled — each only knows about `EngineContext`.
- New stages are inserted without changing any existing stage code.
- `rollback()` and `cleanup()` guarantee resource safety on all code paths.

---

## 5. Observer / Event Bus — `MedicalAIEventBus`

### Intent
Allow external modules to hook into orchestration lifecycle events without modifying the engine's source code.

### Structure
```python
class MedicalAIEventBus:
    _subscribers: Dict[str, List[Callable]] = defaultdict(list)
    _lock = threading.Lock()

    @classmethod
    def subscribe(cls, event: str, handler: Callable[[dict], None]) -> None:
        with cls._lock:
            cls._subscribers[event].append(handler)

    @classmethod
    def publish(cls, event: str, payload: dict) -> None:
        for handler in cls._subscribers.get(event, []):
            handler(payload)

    @classmethod
    def unsubscribe(cls, event: str, handler: Callable) -> None:
        with cls._lock:
            cls._subscribers[event].remove(handler)
```

### Why It's Used
- Module 15 (Grad-CAM) subscribes to `AfterInference` — no Module 14 source changes required.
- Module 19 (Database) subscribes to `AfterPrediction` to persist results.
- Monitoring and telemetry tools can hook in via `BeforePrediction` / `AfterPrediction`.

---

## 6. Context Object Pattern — `EngineContext`

### Intent
Encapsulate all request state, intermediate results, logs, metrics, and extension data in a single object passed to every pipeline stage — eliminating parameter sprawl.

### Why It's Used
- No stage signature needs more than `process(self, ctx: EngineContext)`.
- `ctx.future_extensions` dict absorbs data from future modules without schema changes.
- `ctx.timestamps` enables nanosecond-level per-stage profiling.
- `ctx.logs` and `ctx.errors` provide a complete per-request audit trail.

---

## 7. Plugin Architecture

### Intent
Allow future disease modality engines to register and activate without modifying any existing orchestration code.

### How It Works (Python Entry Points)
```toml
# pyproject.toml for future Module 24 (Mammography)
[project.entry-points."medai.adapters"]
mammography = "mammography.adapter:MammographyPredictorAdapter"
```

At startup, the orchestrator discovers all registered entry points and auto-registers them:
```python
import importlib.metadata
for ep in importlib.metadata.entry_points(group="medai.adapters"):
    adapter_cls = ep.load()
    ModalityEngineRegistry.register(ep.name, adapter_cls)
```

### Why It's Used
- Completely decoupled module integration — no imports needed in core orchestrator.
- Enables third-party modality extensions.

---

## 8. Null Object Pattern — Disabled Optional Stages

### Intent
Keep optional pipeline stages (GradCAM, Anomaly Detection) in the pipeline at all times, but implement them as no-ops when their feature flag is disabled — avoiding conditional branches in the orchestrator core.

### Structure
```python
class GradCAMHookStage(BasePipelineStage):
    @property
    def name(self): return "GradCAMHookStage"

    def process(self, ctx: EngineContext) -> None:
        if not ctx.future_extensions.get("gradcam_enabled", False):
            ctx.logs.append("[GradCAMHookStage] Disabled — skipped.")
            return
        # ... actual Grad-CAM logic (Module 15)
```

### Why It's Used
- The pipeline always runs 12 stages in the same fixed order — no conditional branching in `OrchestrationPipeline.run()`.
- Disabling a feature is a configuration change, not a code change.

---

*Design Patterns Reference — Medical AI Orchestration Engine (Module 14)*  
*8 Patterns Documented — Version 1.1 — August 1, 2026*
