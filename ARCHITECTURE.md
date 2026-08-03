# System Architecture — Medical Image Diagnosis & Anomaly Detection

## 1. Overview

The Medical Image Diagnosis and Anomaly Detection System is a modular, research-oriented, production-quality Artificial Intelligence platform.

```
                           Raw Datasets
                                │
                                ▼
               Module 1: Dataset Discovery & Management
                                │
                                ▼
               Module 2: Metadata Standardization
                                │
                                ▼
               Module 3: Preprocessing Pipeline (224x224)
                                │
                                ▼
              ### Module 16: Anomaly Detection Framework (Module 16.1)
medical_ai_engine/anomaly/
├── interfaces/         ← IDetectionStrategy & IAnomalyRegistry contracts
├── anomaly_configuration.py ← Immutable configuration
├── anomaly_context.py       ← Mutable execution context
├── anomaly_result.py        ← Immutable AnomalyResult & enums
├── anomaly_registry.py      ← Thread-safe strategy plugin registry
├── confidence_analyzer.py   ← Step 1: Confidence trust scorer
├── ood_detector.py          ← Step 2: OOD detector facade
├── unknown_disease_detector.py ← Step 3: Unknown disease detector facade
├── risk_assessor.py         ← Step 4: Multi-signal clinical risk assessor
├── decision_engine.py       ← Step 5: Priority decision & action engine
└── anomaly_detection_engine.py ← Master AnomalyDetectionEngine facade
                                │
                                ▼
              ### Module 15: Explainability Framework (XAI)
medical_ai_engine/explainability/
├── architecture/       ← Core interfaces & context
├── registry/           ← CAM Production Registry & Plugin Loader (15.2)
├── plugins/gradcam/    ← Grad-CAM Engine & Hook Manager (15.3)
├── visualization/      ← Heatmap Renderer & Alpha Blender (15.4)
├── clinical/           ← Clinical Explanation Engine & Report Builder (15.5)
├── integration/        ← Explainability Integrator & Workflow Coordinator (15.6)
└── verification/       ← Master Verification Runner & Benchmark Suite (15.7)
                                │
                                ▼
               Module 4: Augmentation Framework (Offline)
                                │
                                ▼
               Module 5: Exploratory Data Analysis (EDA)
                                │
                                ▼
               Module 5.5: System Verification (GO/NO-GO)
                                │
                                ▼
           ┌─────────────────────────────────────────┐
           │ Module 6: Modality Classifier Engine    │
           └─────────────────────────────────────────┘
                                │
       ┌──────────┬─────────────┼─────────────┬──────────┬──────────┬──────────┐
       ▼          ▼             ▼             ▼          ▼          ▼          ▼
   Module 7    Module 8      Module 9    Module 10  Module 11  Module 12  Module 13
   Brain MRI   Chest X-Ray  Brain CT      Histo     Retina      Skin    Ultrasound
       │          │             │             │          │          │          │
       └──────────┴─────────────┼─────────────┴──────────┴──────────┴──────────┘
                                ▼
           ┌─────────────────────────────────────────────────────┐
           │ Module 14: Unified Medical AI Orchestration Engine  │
           │ (Registry · Factory · Strategy · Pipeline · Plugin) │
           └─────────────────────────────────────────────────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
       Module 15          Module 16          Module 18
       Grad-CAM            Anomaly         Report Generator
    (Explainable AI)     Detection           (NLP/LLM)
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
       Module 19          Module 20          Module 21
       Database           Gradio UI       Docker + FastAPI
       & Logging        (Web Interface)    (REST Deployment)
```

---

## 2. Phase Status

### Phase 1: Data Engineering Pipeline (COMPLETE)
| Module | Name | Status |
|---|---|---|
| Module 1 | Dataset Discovery & Management | ✔ Complete |
| Module 2 | Metadata Standardization & Management | ✔ Complete |
| Module 3 | Medical Image Preprocessing Framework | ✔ Complete |
| Module 4 | Medical Image Augmentation Framework | ✔ Complete |
| Module 5 | Medical Dataset EDA | ✔ Complete |
| Module 5.5 | System Verification & QA | ✔ Complete (GO Decision) |

### Phase 2: AI Modality Engines (COMPLETE)
| Module | Name | Status |
|---|---|---|
| Module 6 | Modality Classification Framework | ✔ Complete |
| Module 7 | Brain MRI Disease Analysis Framework | ✔ Complete (Reference Template) |
| Module 8 | Chest X-ray Disease Analysis Framework | ✔ Complete |
| Module 9 | Brain CT Image Analysis Framework | ✔ Complete |
| Module 10 | Histopathology Cancer Subtyping Framework | ✔ Complete |
| Module 11 | Retina Disease Analysis Framework | ✔ Complete |
| Module 12 | Skin Lesion Cancer Classification Framework | ✔ Complete |
| Module 13 | Ultrasound Disease Analysis Framework | ✔ Complete |

### Phase 3: Core AI Infrastructure & Integration (IN PROGRESS)
| Module | Name | Status |
|---|---|---|
| Module 14.1 | Medical AI Platform Architecture Design | ✔ Complete — Blueprint in `docs/module14_architecture_design.md` |
| Module 14.2 | Unified Prediction Engine & Modality Router | ⏳ Awaiting Approval |
| Module 15 | Explainable AI (Grad-CAM & Attention Maps) | Planned |
| Module 16 | Anomaly Detection (Unsupervised Autoencoders) | Planned |
| Module 17 | Global Model Evaluation & Benchmarking | Planned |
| Module 18 | Medical Report Generation Engine | Planned |
| Module 19 | Prediction History Database & Logging | Planned |

### Phase 4: Deployment & User Interface
| Module | Name | Status |
|---|---|---|
| Module 20 | Gradio Web Application User Interface | Planned |
| Module 21 | Deployment & Containerization (Docker + FastAPI REST) | Planned |
| Module 22 | Integration Testing & QA Suite | Planned |
| Module 23 | Documentation & User Manual | Planned |

---

## 3. Module 14.1 Orchestration Engine Architecture (Blueprint)

See full specification: [`docs/module14_architecture_design.md`](docs/module14_architecture_design.md)

### Design Patterns Used
1. **Registry Pattern** — `ModalityEngineRegistry`: Dynamic registration of all 7 disease module adapters.
2. **Factory Pattern** — `PredictorAdapterFactory`: Instantiates the correct adapter based on routed modality.
3. **Strategy Pattern** — `InferenceStrategy`: Interchangeable execution backends (PyTorch/ONNX/TorchScript).
4. **Pipeline Pattern** — `OrchestrationPipeline`: 6-stage sequential execution (Validate → Route → Select → Infer → Normalize → Report).
5. **Plugin Architecture**: Future modalities register without touching core engine source code.

### Component Hierarchy
```
MedicalAIOrchestrator
├── SchemaValidator
├── ModalityRouter  (wraps Module 6)
├── ModalityEngineRegistry
│   ├── BrainMRIPredictorAdapter    (wraps Module 7)
│   ├── ChestXRayPredictorAdapter   (wraps Module 8)
│   ├── BrainCTPredictorAdapter     (wraps Module 9)
│   ├── HistopathologyPredictorAdapter (wraps Module 10)
│   ├── RetinaPredictorAdapter      (wraps Module 11)
│   ├── SkinPredictorAdapter        (wraps Module 12)
│   └── UltrasoundPredictorAdapter  (wraps Module 13)
├── PredictorAdapterFactory
├── ReliabilityAndSeverityNormalizer
└── ClinicalReportAggregator
```

---

## 4. Shared Architecture Pattern (Modules 7–13)

All disease-specific modules follow the same 9-sub-package structure:
```
<module>/
├── utils/      (config.py, logger.py, checkpoint_manager.py)
├── datasets/   (dataset_loader.py, dataset_builder.py)
├── models/     (backbone_factory.py, <module>_model.py)
├── training/   (trainer.py, callbacks.py, metrics.py)
├── evaluation/ (evaluator.py, roc.py, confusion_matrix.py, misclassified.py, benchmark.py)
├── reports/    (clinical_report.py, performance_report.py)
├── inference/  (predict_<module>.py, batch_predict.py)
└── export/     (onnx_export.py, torchscript_export.py)
```

---

## 5. Shared Components (Zero Duplication)

All Phase 2 modules (7–13) reuse from Module 6 (`modality_classifier`):
- `BackboneFactory.create_backbone()` — backbone instantiation (EfficientNet, ResNet, DenseNet)
- `ModularClassifierHead` — GAP → BatchNorm → Dropout → Linear → Softmax
- `EarlyStopping`, `ModelCheckpoint` — training callbacks
- `compute_classification_metrics()` — accuracy, precision, recall, macro/weighted F1
- `IMAGENET_MEAN`, `IMAGENET_STD` — normalization constants
- `setup_logger()` — centralized file + stream logging
