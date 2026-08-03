# Module Interfaces — Medical AI Orchestration Engine (Module 14)

> **Module:** 14 — Unified Medical AI Orchestration Engine  
> **Specification Version:** 1.1  
> **Last Updated:** August 1, 2026  

---

## 1. `BasePredictorAdapter` — Modality Adapter Interface

Every disease module adapter must implement this interface.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from PIL import Image


class BasePredictorAdapter(ABC):
    """
    Abstract Base Class for all modality predictor adapters.

    Each adapter wraps one disease-specific module (7, 8, 9, 10, 11, 12, or 13)
    and exposes a uniform predict() interface to the orchestrator.

    Registered adapters:
        "brain_mri"      → BrainMRIPredictorAdapter     (Module 7)
        "chest_xray"     → ChestXRayPredictorAdapter    (Module 8)
        "brain_ct"       → BrainCTPredictorAdapter      (Module 9)
        "histopathology" → HistopathologyPredictorAdapter (Module 10)
        "retina"         → RetinaPredictorAdapter        (Module 11)
        "skin"           → SkinPredictorAdapter          (Module 12)
        "ultrasound"     → UltrasoundPredictorAdapter    (Module 13)
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the lowercase modality key, e.g. 'skin', 'retina'."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights have been loaded into memory."""
        return hasattr(self, "_model") and self._model is not None

    @abstractmethod
    def load_model(self, checkpoint_path: str, label_map_path: str) -> None:
        """
        Load model weights and class label mapping from disk.

        Args:
            checkpoint_path: Path to the .pt model checkpoint.
            label_map_path:  Path to the JSON label map file.

        Raises:
            ModelLoadError: If checkpoint or label map cannot be read.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Execute disease prediction on a PIL Image.

        Args:
            image: A 224x224 or larger PIL RGB image.

        Returns:
            A dict conforming to the module prediction schema.
            Required keys:
                module, modality, predicted_class, confidence,
                top3_predictions, probability_distribution,
                model_version, inference_time_ms, timestamp, status

            Module-specific optional keys:
                severity_grade     (Module 11 - retina)
                prediction_reliability, requires_manual_review  (Modules 12, 13)

        Raises:
            InferenceError: If the forward pass fails.
        """
        pass

    def extract_features(self, image: Image.Image) -> Optional[Any]:
        """
        Extract intermediate feature maps for Grad-CAM (Module 15).

        Default: Not supported. Subclasses override to expose feature activations.

        Returns:
            Feature tensor or activation map, or None if not implemented.
        """
        return None

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "BasePredictorAdapter":
        """
        Instantiate adapter from configuration dict.
        Called by PredictorAdapterFactory.

        Args:
            config: Dict with keys 'checkpoint' and 'label_map'.

        Returns:
            Initialized (but not yet loaded) adapter instance.
        """
        instance = cls()
        if config:
            instance._checkpoint_path = config.get("checkpoint")
            instance._label_map_path = config.get("label_map")
        return instance
```

---

## 2. `BasePipelineStage` — Orchestration Stage Interface

Every orchestration stage implements this interface.

```python
from abc import ABC, abstractmethod


class BasePipelineStage(ABC):
    """
    Abstract interface for all orchestration pipeline stages.

    Lifecycle per stage:
        validate(ctx)  → Pre-condition assertions (raises on failure)
        process(ctx)   → Primary logic (mutates ctx in place)
        rollback(ctx)  → Undo partial mutations (called on exception)
        cleanup(ctx)   → Resource teardown (always called via finally block)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage identifier for logging and diagnostics."""
        pass

    def validate(self, ctx: "EngineContext") -> None:
        """
        Assert pre-conditions before process() runs.
        Default: no-op. Override when pre-conditions exist.
        Raises PipelineStageValidationError on failure.
        """
        pass

    @abstractmethod
    def process(self, ctx: "EngineContext") -> None:
        """
        Primary stage logic. Mutates ctx in place.

        Responsibilities:
            - Record start time in ctx.timestamps[self.name + "_start"]
            - Record end time in ctx.timestamps[self.name + "_end"]
            - Append log entries to ctx.logs
            - Record timing in ctx.performance_metrics[self.name + "_ms"]
            - Populate ctx.errors (list) on non-fatal failures
        """
        pass

    def rollback(self, ctx: "EngineContext") -> None:
        """
        Undo any partial side effects if process() raised.
        Default: no-op. Stateless stages need not override.
        """
        pass

    def cleanup(self, ctx: "EngineContext") -> None:
        """
        Release resources held during process().
        Always called, even on exception.
        Default: no-op.
        """
        pass
```

---

## 3. `ModalityEngineRegistry` Interface

```python
class ModalityEngineRegistry:
    """
    Thread-safe singleton registry of all modality predictor adapters.
    Supports registration, lazy loading, caching, reload, and hot-swap.
    """

    @classmethod
    def register(cls, modality: str, adapter_cls: Type[BasePredictorAdapter]) -> None:
        """Register an adapter class for a modality key (does not load weights)."""
        ...

    @classmethod
    def get(cls, modality: str) -> BasePredictorAdapter:
        """
        Return a loaded, cached adapter for the given modality.
        Lazy-loads model weights on first access per modality.
        Thread-safe.
        """
        ...

    @classmethod
    def get_class(cls, modality: str) -> Optional[Type[BasePredictorAdapter]]:
        """Return the registered adapter class without instantiating it."""
        ...

    @classmethod
    def is_registered(cls, modality: str) -> bool:
        """Returns True if the modality has a registered adapter class."""
        ...

    @classmethod
    def is_loaded(cls, modality: str) -> bool:
        """Returns True if the modality adapter is loaded and cached."""
        ...

    @classmethod
    def registered_modalities(cls) -> List[str]:
        """Return list of all registered modality keys."""
        ...

    @classmethod
    def reload(cls, modality: str) -> None:
        """Evict cached adapter instance. Forces fresh load on next get()."""
        ...

    @classmethod
    def swap(cls, modality: str, new_checkpoint_path: str) -> None:
        """
        Zero-downtime hot-swap of model weights for a modality.
        Loads new weights before atomically replacing the cached instance.
        """
        ...
```

---

## 4. `MedicalAIEventBus` Interface

```python
class MedicalAIEventBus:
    """
    Thread-safe in-process publish/subscribe event bus.
    Used for lifecycle event broadcasting without coupling.
    """

    EVENTS = [
        "BeforePrediction",
        "AfterPrediction",
        "BeforeRouting",
        "AfterRouting",
        "BeforeInference",
        "AfterInference",
        "BeforeReport",
        "AfterReport",
        "OnError",
        "OnLowConfidence",
    ]

    @classmethod
    def subscribe(cls, event: str, handler: Callable[[dict], None]) -> None:
        """Register a handler for a named lifecycle event."""
        ...

    @classmethod
    def publish(cls, event: str, payload: dict) -> None:
        """
        Fire all handlers registered for event.
        Handlers are called synchronously in subscription order.
        """
        ...

    @classmethod
    def unsubscribe(cls, event: str, handler: Callable) -> None:
        """Remove a previously registered handler."""
        ...

    @classmethod
    def clear(cls, event: Optional[str] = None) -> None:
        """
        Remove all handlers for a specific event, or all events if None.
        Primarily used in test teardown.
        """
        ...
```

---

## 5. `MedicalAIOrchestrator` — Public API Interface

```python
class MedicalAIOrchestrator:
    """
    Public entry point for the Unified Medical AI Orchestration Engine.
    Integrates Modules 1–13 into a single prediction platform.
    """

    def __init__(self, config_path: str = "configs/orchestration.yaml") -> None:
        """Load configuration, initialize registry, build pipeline."""
        ...

    def predict(
        self,
        image_source: Union[str, Path, Image.Image, bytes, np.ndarray],
        manual_modality_override: Optional[str] = None,
        top_k: int = 3,
        inference_backend: str = "pytorch",
        enable_clinical_report: bool = True,
    ) -> "UnifiedPredictionResponse":
        """
        Execute the full 12-stage orchestration pipeline on a single image.

        Args:
            image_source: File path, PIL Image, raw bytes, or numpy array.
            manual_modality_override: Skip modality detection; use this modality key.
            top_k: Number of top predictions to return.
            inference_backend: One of "pytorch", "onnx", "torchscript".
            enable_clinical_report: If True, assembles clinical diagnostic notes.

        Returns:
            UnifiedPredictionResponse dataclass (JSON-serializable).

        Raises:
            ImageValidationError: On unreadable or corrupt input.
            UnsupportedModalityError: On unregistered modality.
        """
        ...

    def batch_predict(
        self,
        image_directory: Union[str, Path],
        output_path: Optional[str] = None,
        **predict_kwargs,
    ) -> List["UnifiedPredictionResponse"]:
        """
        Run predict() on all images in a directory.
        Optionally saves results to a JSON/CSV file.
        """
        ...

    def warm_up(self, modalities: Optional[List[str]] = None) -> None:
        """
        Pre-load model weights for specified modalities (or all if None).
        Eliminates cold-start latency on first request per modality.
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """
        Return current registry status: registered modalities, loaded adapters, device.
        Useful for health checks and monitoring.
        """
        ...
```

---

## 6. Adapter Implementation Reference: `SkinPredictorAdapter`

Illustrative example of how an adapter wraps an existing module predictor.

```python
from skin.inference.predict_skin import SkinPredictor
from skin.utils.config import SkinConfig
from orchestrator.adapters.base import BasePredictorAdapter


class SkinPredictorAdapter(BasePredictorAdapter):
    """
    Adapter wrapping Module 12 (Skin Lesion Cancer Classification).
    Exposes Module 12 predictions via BasePredictorAdapter interface.
    No modification to Module 12 source code.
    """

    @property
    def modality(self) -> str:
        return "skin"

    def load_model(self, checkpoint_path: str, label_map_path: str) -> None:
        config = SkinConfig()
        config.checkpoint_path = checkpoint_path
        config.label_map_path = label_map_path
        self._predictor = SkinPredictor(config)
        self._predictor.load()

    def predict(self, image: Image.Image) -> dict:
        raw = self._predictor.predict(image)
        # raw dict already conforms to Module 14 schema
        # (contains: module, modality, predicted_class, confidence,
        #  prediction_reliability, requires_manual_review, top3_predictions, etc.)
        return raw
```

---

*Module Interfaces Specification — Medical AI Orchestration Engine (Module 14)*  
*Version 1.1 — 6 Interface Definitions — August 1, 2026*
