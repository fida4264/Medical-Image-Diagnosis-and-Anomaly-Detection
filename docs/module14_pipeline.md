# Module 14.2.4: Orchestration Pipeline Stages

> **Module Version:** `0.14.2.4`  
> **Package Location:** `medical_ai_engine/pipeline/`  
> **Status:** OPERATIONAL & TESTED  
> **Date:** August 2, 2026  

---

## 1. Purpose

Module 14.2.4 implements the **Orchestration Pipeline Stages & PipelineExecutor** for the Medical AI Platform.

It coordinates the complete prediction request lifecycle through 9 modular, sequential `BasePipelineStage` implementations. Every stage receives, inspects, and mutates the **same shared `EngineContext` instance** without creating independent context objects or introducing direct disease model dependencies.

---

## 2. Architecture Overview

```
MedicalAIEngine Core
        │
        ▼
 PipelineExecutor  ──► EngineEventPublisher (StageStartedEvent, StageCompletedEvent, etc.)
        │
        ├──► 1. InputValidationStage           (validates source, decodes PIL Image, size/mode metadata)
        ├──► 2. ImageQualityAssessmentStage     (computes resolution & quality scores)
        ├──► 3. ModalityClassificationStage    (invokes ModalityRouter / Module 6)
        ├──► 4. RoutingStage                   (evaluates outcome, checks ModelRegistry adapter)
        ├──► 5. AdapterPredictionStage         (executes IPredictorAdapter.predict())
        ├──► 6. ConfidenceAssessmentStage      (normalizes confidence, reliability, manual review)
        ├──► 7. ClinicalReportHookStage        (generates diagnostic report structure)
        ├──► 8. ResponsePreparationStage       (assembles prediction JSON result)
        └──► 9. LoggingStage                   (records execution audit summary)
        │
        ▼
 Populated EngineContext (COMPLETED)
```

---

## 3. The 9 Pipeline Stages

| Order | Stage Class Name | Input Context State | Output Context State | Key Action |
|---|---|---|---|---|
| 1 | `InputValidationStage` | `image_source`, `image_bytes`, or `image` | `image` (PIL Image), `metadata['format']`, `metadata['size']`, `metadata['color_mode']` | Decodes image & validates input parameters |
| 2 | `ImageQualityAssessmentStage` | `image` | `metadata['quality_score']` | Assesses resolution & image clarity heuristics |
| 3 | `ModalityClassificationStage` | `image` or `manual_modality_override` | `modality_prediction` dict | Invokes `ModalityRouter` / Module 6 |
| 4 | `RoutingStage` | `modality_prediction` | `metadata['selected_adapter']` | Verifies `ROUTE_TO_MODEL` & `ModelRegistry` adapter |
| 5 | `AdapterPredictionStage` | `image`, `metadata['selected_adapter']` | `disease_prediction` dict | Runs disease module `IPredictorAdapter.predict()` |
| 6 | `ConfidenceAssessmentStage` | `disease_prediction` | `confidence` dict | Computes reliability (`high`/`moderate`/`low`) & review flags |
| 7 | `ClinicalReportHookStage` | `disease_prediction`, `confidence` | `clinical_report` dict | Assembles structured diagnostic report |
| 8 | `ResponsePreparationStage` | All prediction outputs | `prediction_result` dict | Prepares normalized JSON prediction object |
| 9 | `LoggingStage` | `EngineContext` | Log entry & audit timestamps | Records diagnostic execution summary |

---

## 4. Shared EngineContext Model

- **Zero Duplicate Contexts:** No stage creates its own context object or returns a new context.
- **Reference Passing:** The identical `EngineContext` instance is passed through `stage.process(ctx)` across all 9 stages.
- **Progressive State Mutation:** Each stage populates its designated fields while preserving upstream fields for downstream stages.

---

## 5. Rollback Strategy in Reverse Order

If any stage fails during `validate(ctx)` or `process(ctx)`:
1. The failure is caught by `PipelineExecutor`.
2. The error message is appended to `ctx.add_error(stage_name, err_msg)`.
3. `StageFailedEvent` is published to the event bus.
4. `PipelineExecutor` iterates over all **already completed stages in REVERSE order** and invokes `stage.rollback(ctx)`.
5. For each rolled-back stage, `StageRollbackEvent` is published.
6. `stage.cleanup(ctx)` is invoked on all processed/attempted stages in reverse order.
7. `PipelineExecutionException` is raised to signal failure to the engine core.

---

## 6. Timing & Performance Metrics Model

- Each stage is wrapped in `EngineTimer()`.
- Per-stage execution duration in milliseconds is stored in `ctx.performance_metrics["<StageName>_ms"]`.
- Total pipeline wall-clock time is recorded in `ctx.performance_metrics["total_pipeline_ms"]`.

---

## 7. Event Bus Integration

The `PipelineExecutor` emits granular stage events via `EngineEventPublisher`:

- **`StageStartedEvent`**: Published immediately before `stage.process(ctx)`.
- **`StageCompletedEvent`**: Published upon successful completion with `duration_ms`.
- **`StageFailedEvent`**: Published if stage execution or validation raises an exception.
- **`StageRollbackEvent`**: Published during reverse-order rollback.

---

## 8. Exception Hierarchy

```
EngineException (ENG-000)
    └── EngineExecutionException (ENG-300)
          └── PipelineExecutionException (ENG-320)
                ├── StageValidationException (ENG-321)
                └── StageRollbackException (ENG-322)
```

---

## 9. Future Extension Points

- **Grad-CAM Hook Stage (Module 15):** Can be inserted between `AdapterPredictionStage` and `ConfidenceAssessmentStage` to generate explainability heatmaps into `ctx.gradcam_result`.
- **Anomaly Detection Hook Stage (Module 16):** Can run in parallel with disease prediction to populate `ctx.anomaly_result`.
- **Database Persistence Stage (Module 19):** Can be appended after `LoggingStage` to store prediction records into SQL/NoSQL databases.

---

*Module 14.2.4: Orchestration Pipeline Stages Documentation — v0.14.2.4*
