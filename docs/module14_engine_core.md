# Module 14.2.1: MedicalAIEngine Core — Operating System Specification

> **Module Version:** `0.14.2.1`  
> **System Status:** OPERATIONAL & TESTED  
> **Package Location:** `medical_ai_engine/`  
> **Author:** Antigravity AI Engineering Team  
> **Date:** August 1, 2026  

---

## 1. Purpose & Core Philosophy

**Module 14.2.1 (`medical_ai_engine`)** constructs the foundational **Engine Core** for the Medical AI Platform.

It acts as the central **Operating System** of the platform:
- Manages request lifecycle state machine (`CREATED` → `INITIALIZED` → `RUNNING` → `COMPLETED` / `FAILED` / `CANCELLED`).
- Creates and passes `EngineContext` immutably by reference through pipeline stage interfaces.
- Tracks `ExecutionSession` metadata, thread ownership, and execution duration.
- Publishes lifecycle events (`EngineStarted`, `EngineInitialized`, `EngineRunning`, `EngineCompleted`, `EngineFailed`, `EngineCancelled`).
- Provides thread-safe operational metrics (`EngineStatistics`).
- Enforces strict error handling and logging (`logs/medical_ai_engine.log`).

> **Core Isolation Principle:**
> The Engine Core **does NOT import or depend on any AI modality models** (Modules 6–13). It contains zero model routing or prediction logic. It provides the pure execution environment through which all future orchestration components execute.

---

## 2. Package Structure & File Inventory

```
medical_ai_engine/
├── __init__.py                   # Core package exports
├── core/
│   ├── __init__.py
│   ├── medical_ai_engine.py      # MedicalAIEngine core orchestrator class
│   ├── engine_context.py        # EngineContext dataclass
│   ├── engine_state.py          # EngineState enum & EngineLifecycle state machine
│   ├── execution_session.py     # ExecutionSession tracking class
│   ├── engine_configuration.py  # EngineConfiguration dataclass & YAML loader
│   ├── engine_statistics.py     # EngineStatistics thread-safe metrics counter
│   └── exceptions.py            # EngineException hierarchy
├── interfaces/
│   ├── __init__.py
│   ├── engine_interface.py      # IEngine interface
│   ├── pipeline_stage.py        # BasePipelineStage interface
│   ├── event_publisher.py       # IEventPublisher interface
│   ├── logger_interface.py      # IEngineLogger interface
│   └── config_interface.py      # IEngineConfiguration interface
├── events/
│   ├── __init__.py
│   ├── engine_events.py         # BaseEngineEvent & EngineEventPublisher
│   └── lifecycle_events.py      # Concrete lifecycle event classes
├── utils/
│   ├── __init__.py
│   ├── request_id.py            # UUID request & session ID generators
│   ├── timer.py                 # EngineTimer high-precision monotonic timer
│   ├── state_manager.py         # EngineStateManager lifecycle tracker
│   ├── thread_manager.py        # EngineThreadManager worker thread pool
│   └── health_monitor.py        # EngineHealthMonitor system monitor
├── tests/
│   └── test_engine_core.py      # Comprehensive 13-test PyTest suite
├── docs/
│   └── module14_engine_core.md  # This specification document
└── logs/
    └── medical_ai_engine.log    # Structured runtime log file
```

---

## 3. Engine Lifecycle & State Machine

The engine and every request session transition through a strict, deterministic state machine:

```
  CREATED ────────► INITIALIZED ────────► RUNNING ────────► COMPLETED (terminal)
     │                     │                │      │
     │                     ▼                │      ▼
     └─────────────────► FAILED ◄───────────┴──► CANCELLED (terminal)
                       (terminal)
```

### Valid State Transitions (`EngineState`)

| Source State | Legal Target States | Description |
|---|---|---|
| `CREATED` | `INITIALIZED`, `FAILED`, `CANCELLED` | Instantiated but configuration not yet verified |
| `INITIALIZED` | `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED` | Configuration verified, thread pool ready |
| `RUNNING` | `WAITING`, `COMPLETED`, `FAILED`, `CANCELLED` | Currently processing pipeline stages |
| `WAITING` | `RUNNING`, `FAILED`, `CANCELLED` | Temporarily paused for external async response |
| `COMPLETED` | *(None — Terminal)* | Execution completed successfully |
| `FAILED` | *(None — Terminal)* | Unhandled exception occurred |
| `CANCELLED` | *(None — Terminal)* | Session cancelled by timeout or user |

Attempting any transition not listed in this table raises `EngineStateException` with error code `ENG-200`.

---

## 4. EngineContext & Data Flow

Every prediction request generates a single `EngineContext` instance that is passed by reference through every `BasePipelineStage`.

```python
@dataclass
class EngineContext:
    request_id: str
    session_id: str
    image: Optional[Any] = None
    image_source: Optional[Union[str, Path]] = None
    image_bytes: Optional[bytes] = None
    manual_modality_override: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, str] = field(default_factory=dict)
    execution_state: str = EngineState.CREATED.value
    modality_prediction: Optional[Dict[str, Any]] = None
    disease_prediction: Optional[Dict[str, Any]] = None
    prediction_result: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[str, Any]] = None
    clinical_report: Optional[Dict[str, Any]] = None
    gradcam_result: Optional[Dict[str, Any]] = None
    anomaly_result: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=dict)
    future_extensions: Dict[str, Any] = field(default_factory=dict)
```

No stage signature requires more than `process(self, ctx: EngineContext) -> None`.

---

## 5. ExecutionSession & Session Tracking

`ExecutionSession` wraps a single execution request, tracking:
- Nanosecond start/end timestamps via `time.monotonic()`
- Request correlation (`request_id` and `session_id`)
- Thread and process ownership (`process_id`, `thread_id`, `thread_name`)
- Execution duration calculations (`duration_ms()`)
- Lifecycle state transitions (`transition_state()`)

---

## 6. Event System & Lifecycle Publication

`EngineEventPublisher` publishes structured lifecycle events:
- `EngineStartedEvent`: Fired when `MedicalAIEngine` is constructed.
- `EngineInitializedEvent`: Fired when `initialize()` completes.
- `EngineRunningEvent`: Fired when `execute()` starts a session.
- `EngineCompletedEvent`: Fired when a request completes successfully.
- `EngineFailedEvent`: Fired when a request raises an exception.
- `EngineCancelledEvent`: Fired when a session is cancelled.

---

## 7. EngineStatistics & Thread Safety

`EngineStatistics` provides thread-safe counters protected by an `RLock`:
- `total_executions`: Cumulative request count
- `active_sessions`: Currently running sessions
- `completed_sessions`: Successfully finished sessions
- `failed_sessions`: Errored sessions
- `average_execution_time_ms`: Moving average latency
- `average_initialization_time_ms`: Moving average init latency
- `success_rate_percent`: Percentage of successful executions
- `uptime_seconds`: Engine running time

---

## 8. Configuration Specification (`configs/orchestration_engine.yaml`)

```yaml
engine:
  name: "MedicalAIEngineCore"
  version: "0.14.2.1"
  environment: "production"

execution:
  request_timeout_seconds: 30.0
  max_concurrent_threads: 16
  thread_pool_prefix: "MedAI-EngineWorker"

logging:
  level: "INFO"
  file_path: "logs/medical_ai_engine.log"
  enable_console: true
  enable_file: true

health:
  check_interval_seconds: 60.0
  statistics_interval_seconds: 300.0
  enable_monitor: true

async_distributed:
  enable_async: false
  enable_distributed: false
  node_id: "node_local_01"
```

---

## 9. Exception Hierarchy

```
EngineException (ENG-000)
├── EngineInitializationException (ENG-100)
│   └── ConfigurationException (ENG-101)
├── EngineStateException (ENG-200)
├── EngineExecutionException (ENG-300)
│   ├── ExecutionTimeoutException (ENG-301)
│   └── SessionNotFoundException (ENG-302)
├── ContextException (ENG-400)
├── HealthCheckException (ENG-500)
├── EventPublishException (ENG-600)
└── RegistryException (ENG-700)
```

---

## 10. Future Integration Blueprints

Every subsequent submodule connects to this Engine Core without modifying `medical_ai_engine/core/`:

| Future Submodule | Integration Point |
|---|---|
| **Module 14.2.2 (Registry)** | Injected into `EngineContext.metadata["selected_adapter"]` via a `BasePipelineStage` |
| **Module 14.2.3 (Router)** | Injected into `EngineContext.modality_prediction` via a `DetectModalityStage` |
| **Module 14.2.4 (Pipeline)** | Passed to `MedicalAIEngine(pipeline_stages=[...])` at engine instantiation |
| **Module 14.2.5 (Response Builder)** | Populates `EngineContext.prediction_result` via an `AssembleResponseStage` |
| **Module 15 (Grad-CAM)** | Populates `EngineContext.gradcam_result` via a `GradCAMHookStage` |
| **Module 16 (Anomaly Detection)** | Populates `EngineContext.anomaly_result` via an `AnomalyHookStage` |
| **Module 20 (Gradio UI)** | Calls `engine.execute(image_source)` directly |
| **Module 21 (Docker REST API)** | FastAPI POST `/v1/predict` calls `engine.execute(bytes)` |

---

*Module 14.2.1: MedicalAIEngine Core Operating System Specification*  
*Version 0.14.2.1 — August 1, 2026*
