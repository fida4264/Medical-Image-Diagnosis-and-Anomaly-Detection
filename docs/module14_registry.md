# Module 14.2.2: Model Registry & Adapter Framework

> **Module Version:** `0.14.2.2`  
> **Package Location:** `medical_ai_engine/registry/` & `medical_ai_engine/adapters/`  
> **Status:** OPERATIONAL & TESTED  
> **Date:** August 1, 2026  

---

## 1. Purpose

Module 14.2.2 implements the **Model Registry & Adapter Framework** — the single source of truth for all available disease AI models in the Medical AI Platform.

The Engine Core (`MedicalAIEngine`) remains completely unaware of individual disease modules. It communicates exclusively with the `ModelRegistry`, which returns `IPredictorAdapter` instances exposing a uniform `predict()` interface regardless of the underlying module (7–13).

---

## 2. Architecture Overview

```
MedicalAIEngine (Engine Core)
        │
        ▼
  ModelRegistry  ─── AdapterRegistry (class registrations + metadata)
        │         ─── AdapterCache    (loaded instance cache)
        │         ─── AdapterLoader   (model weight loading)
        │         ─── AdapterFactory  (adapter instantiation)
        │
  IPredictorAdapter (uniform interface)
        │
  ┌─────┼─────────────────────────────────┐
  │     │                                 │
BrainMRI ChestXray BrainCT Histopath Retina Skin Ultrasound
(Mod 7) (Mod 8)  (Mod 9) (Mod 10) (Mod 11) (Mod 12) (Mod 13)
```

---

## 3. Registry Lifecycle

```
register_adapter(modality, cls)  →  AdapterRegistry stores class + metadata
        │
get_adapter(modality)             →  Cache hit? → return cached instance
        │                             Cache miss → AdapterLoader.load()
        │                                          → adapter.initialize()
        │                                          → adapter.load_model()
        │                                          → AdapterCache.put()
        │                                          → return instance
invalidate_cache(modality)        →  AdapterCache.invalidate() + shutdown()
reload_adapter(modality)          →  invalidate + get_adapter (fresh load)
hot_swap(modality, new_ckpt)      →  Load new weights → atomic cache replace
```

---

## 4. IPredictorAdapter Interface Contract

Every disease module adapter must implement:

| Method | Signature | Purpose |
|---|---|---|
| `modality` | `@property → str` | Return lowercase modality key |
| `is_loaded` | `@property → bool` | True if model weights in memory |
| `initialize()` | `(config: dict) → None` | Configure paths (no heavy ops) |
| `load_model()` | `(ckpt: str, lbl: str) → None` | Load weights into memory |
| `predict()` | `(image: PIL.Image) → dict` | Execute inference |
| `health_check()` | `() → dict` | Return adapter health status |
| `get_metadata()` | `() → dict` | Return supported classes, version, status |
| `shutdown()` | `() → None` | Unload weights and free memory |

---

## 5. Registered Adapters (Phase 2 Disease Modules)

| Modality Key | Adapter Class | Wraps Module | Supported Classes |
|---|---|---|---|
| `brain_mri` | `BrainMRIPredictorAdapter` | Module 7 | glioma, meningioma, notumor, pituitary |
| `chest_xray` | `ChestXRayPredictorAdapter` | Module 8 | normal, pneumonia, covid |
| `brain_ct` | `BrainCTPredictorAdapter` | Module 9 | head_ct_scan |
| `histopathology` | `HistopathologyPredictorAdapter` | Module 10 | colon_aca, colon_n, lung_aca, lung_n, lung_scc |
| `retina` | `RetinaPredictorAdapter` | Module 11 | No_DR, Mild, Moderate, Severe, Proliferative_DR |
| `skin` | `SkinPredictorAdapter` | Module 12 | melanoma, melanocytic_nevus, basal_cell_carcinoma, actinic_keratosis, benign_keratosis, dermatofibroma, vascular_lesion |
| `ultrasound` | `UltrasoundPredictorAdapter` | Module 13 | benign, malignant, normal |

---

## 6. AdapterCache — Lazy Loading & Memory Management

- Adapters are loaded **only on first request** for a given modality.
- Subsequent requests to the same modality return the **same cached instance** (zero overhead).
- `invalidate_cache(modality)` unloads a specific adapter and calls `shutdown()` to free VRAM/RAM.
- `invalidate_cache(None)` evicts all adapters.

---

## 7. AdapterMetadata — Per-Modality Information Tracking

Each registered modality carries an `AdapterMetadata` object:

| Field | Description |
|---|---|
| `modality` | Lowercase modality key |
| `supported_classes` | List of disease class names |
| `model_version` | Versioned model release string |
| `framework` | PyTorch / ONNX / TorchScript |
| `creation_date` | ISO 8601 UTC registration timestamp |
| `status` | `REGISTERED` / `LOADED` / `ERRORED` / `UNLOADED` |
| `health` | `HEALTHY` / `UNHEALTHY` / `UNKNOWN` |
| `model_path` | Checkpoint path on disk |
| `label_map_path` | JSON label map path on disk |
| `export_formats` | Available export formats |

---

## 8. Thread Safety

All registry operations are protected by `threading.RLock()`:
- `AdapterRegistry._lock` guards class registrations and metadata mutations.
- `AdapterCache._lock` guards cache put/get/invalidate operations.
- `ModelRegistry._lock` guards end-to-end `get_adapter()` transactions to prevent race conditions on first load.

Concurrent reads after first load are fast and contention-free.

---

## 9. Failure Handling

| Scenario | Exception Raised |
|---|---|
| Unregistered modality lookup | `AdapterNotFoundException (REG-102)` |
| Duplicate registration without overwrite | `DuplicateAdapterRegistrationException (REG-101)` |
| Model weight loading failure | `AdapterLoadException (REG-103)` |
| Cache operation error | `AdapterCacheException (REG-104)` |

---

## 10. Future Hot-Swapping

`ModelRegistry.hot_swap(modality, new_checkpoint_path)`:
1. Loads new weights for the adapter into a fresh instance.
2. **Atomically** replaces the cached pointer under `_lock`.
3. Old adapter instance is discarded.

Zero-downtime model upgrade supported without restarting the engine.

---

## 11. Adding Future Modalities

To register a new modality (e.g. Module 24: Mammography), simply:
1. Create `medical_ai_engine/adapters/mammography_adapter.py` implementing `BasePredictorAdapter`.
2. Call `registry.register_adapter("mammography", MammographyPredictorAdapter)`.
3. No changes to Engine Core (`MedicalAIEngine`), Router, or Response Builder.

---

*Module 14.2.2: Model Registry & Adapter Framework Documentation — v0.14.2.2*
