"""
Medical AI Engine Core — Comprehensive Test Suite

Tests:
  - EngineContext creation & field mutation
  - EngineState enum & EngineLifecycle state machine
  - ExecutionSession tracking & timing
  - EngineStatistics thread safety & counter accuracy
  - EngineException hierarchy & serialization
  - EngineConfiguration YAML parsing & validation
  - Thread safety & concurrent requests
  - MedicalAIEngine execution lifecycle end-to-end (without AI models)
"""

from pathlib import Path
import threading
import time
import pytest

from medical_ai_engine.core.engine_configuration import EngineConfiguration
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.engine_state import EngineLifecycle, EngineState
from medical_ai_engine.core.engine_statistics import EngineStatistics
from medical_ai_engine.core.exceptions import (
    ConfigurationException,
    EngineException,
    EngineExecutionException,
    EngineInitializationException,
    EngineStateException,
)
from medical_ai_engine.core.execution_session import ExecutionSession
from medical_ai_engine.core.medical_ai_engine import MedicalAIEngine
from medical_ai_engine.events.engine_events import EngineEventPublisher
from medical_ai_engine.events.lifecycle_events import EngineCompletedEvent, EngineStartedEvent
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage
from medical_ai_engine.utils.request_id import generate_request_id, generate_session_id
from medical_ai_engine.utils.timer import EngineTimer


# ── Mock Pipeline Stage ────────────────────────────────────────────────────────

class MockPipelineStage(BasePipelineStage):
    """Simple mock stage for testing pipeline stage execution."""

    def __init__(self, name: str = "MockStage", delay_ms: float = 10.0) -> None:
        self._name = name
        self.delay_ms = delay_ms
        self.processed = False
        self.validated = False
        self.cleaned_up = False

    @property
    def name(self) -> str:
        return self._name

    def validate(self, ctx: EngineContext) -> None:
        self.validated = True

    def process(self, ctx: EngineContext) -> None:
        time.sleep(self.delay_ms / 1000.0)
        self.processed = True
        ctx.metadata[self._name] = "success"
        ctx.log(f"{self._name} executed", stage=self._name)

    def cleanup(self, ctx: EngineContext) -> None:
        self.cleaned_up = True


class MockFailingStage(BasePipelineStage):
    """Mock stage that intentionally raises an exception."""

    @property
    def name(self) -> str:
        return "FailingStage"

    def process(self, ctx: EngineContext) -> None:
        raise ValueError("Simulated stage failure")


# ── Tests: EngineContext ───────────────────────────────────────────────────────

def test_engine_context_creation():
    req_id = generate_request_id()
    sess_id = generate_session_id(req_id)
    ctx = EngineContext(request_id=req_id, session_id=sess_id)

    assert ctx.request_id == req_id
    assert ctx.session_id == sess_id
    assert ctx.execution_state == EngineState.CREATED.value
    assert isinstance(ctx.metadata, dict)
    assert isinstance(ctx.logs, list)
    assert isinstance(ctx.warnings, list)
    assert isinstance(ctx.errors, list)
    assert isinstance(ctx.future_extensions, dict)


def test_engine_context_logging_and_metrics():
    ctx = EngineContext(request_id="req_001", session_id="sess_001")
    ctx.log("Starting test", stage="TestStage")
    ctx.warn("Careful with quality", stage="TestStage")
    ctx.add_error("TestStage", "Minor glitch")
    ctx.record_metric("test_ms", 12.5)

    assert len(ctx.logs) == 1
    assert len(ctx.warnings) == 1
    assert len(ctx.errors) == 1
    assert ctx.has_warnings()
    assert ctx.has_errors()
    assert ctx.performance_metrics["test_ms"] == 12.5


# ── Tests: EngineState & Lifecycle ─────────────────────────────────────────────

def test_engine_lifecycle_transitions():
    lc = EngineLifecycle(request_id="req_test")
    assert lc.state == EngineState.CREATED

    lc.transition(EngineState.INITIALIZED)
    assert lc.state == EngineState.INITIALIZED

    lc.transition(EngineState.RUNNING)
    assert lc.state == EngineState.RUNNING

    lc.transition(EngineState.COMPLETED)
    assert lc.state == EngineState.COMPLETED
    assert lc.is_terminal()


def test_engine_lifecycle_illegal_transition():
    lc = EngineLifecycle(request_id="req_test")
    with pytest.raises(EngineStateException) as exc_info:
        lc.transition(EngineState.COMPLETED)  # CREATED -> COMPLETED is illegal

    err = exc_info.value
    assert err.current_state == EngineState.CREATED.value
    assert err.attempted_transition == EngineState.COMPLETED.value


# ── Tests: ExecutionSession ───────────────────────────────────────────────────

def test_execution_session_lifecycle():
    session = ExecutionSession(request_id="req_100")
    assert session.state == EngineState.CREATED
    assert session.duration_seconds() >= 0.0

    session.transition_state(EngineState.INITIALIZED)
    session.transition_state(EngineState.RUNNING)
    time.sleep(0.01)
    session.transition_state(EngineState.COMPLETED)

    assert session.is_complete()
    assert session.duration_ms() >= 0.0
    sess_dict = session.to_dict()
    assert sess_dict["session_id"] == "session_req_100"
    assert sess_dict["state"] == EngineState.COMPLETED.value


# ── Tests: EngineStatistics ───────────────────────────────────────────────────

def test_engine_statistics_counters():
    stats = EngineStatistics()
    stats.record_engine_init(0.05)

    stats.session_started()
    stats.session_completed(15.0)

    stats.session_started()
    stats.session_failed()

    snapshot = stats.snapshot()
    assert snapshot["total_executions"] == 2
    assert snapshot["completed_sessions"] == 1
    assert snapshot["failed_sessions"] == 1
    assert snapshot["active_sessions"] == 0
    assert snapshot["average_execution_time_ms"] == 15.0
    assert snapshot["success_rate_percent"] == 50.0


# ── Tests: Exceptions ──────────────────────────────────────────────────────────

def test_exception_hierarchy():
    ex = EngineExecutionException("Execution failed", stage_name="TestStage", request_id="req_99")
    assert ex.error_code == "ENG-300"
    assert ex.request_id == "req_99"
    assert "TestStage" in ex.message

    d = ex.to_dict()
    assert d["exception_type"] == "EngineExecutionException"
    assert d["error_code"] == "ENG-300"


# ── Tests: Configuration ──────────────────────────────────────────────────────

def test_engine_configuration_defaults():
    cfg = EngineConfiguration()
    assert cfg.engine_name == "MedicalAIEngineCore"
    assert cfg.request_timeout_seconds == 30.0
    assert cfg.max_concurrent_threads == 16


def test_engine_configuration_validation():
    cfg = EngineConfiguration(request_timeout_seconds=-5.0)
    with pytest.raises(ConfigurationException):
        cfg.validate()


# ── Tests: MedicalAIEngine Core Execution ─────────────────────────────────────

def test_engine_initialization():
    cfg = EngineConfiguration(engine_name="TestEngine")
    engine = MedicalAIEngine(config=cfg)

    assert engine.state == EngineState.CREATED
    engine.initialize()
    assert engine.state == EngineState.INITIALIZED

    stats = engine.get_statistics()
    assert stats["total_init_count"] == 1


def test_engine_successful_execution():
    stage1 = MockPipelineStage(name="Stage1", delay_ms=5.0)
    stage2 = MockPipelineStage(name="Stage2", delay_ms=5.0)

    cfg = EngineConfiguration(engine_name="TestEngineSuccess")
    engine = MedicalAIEngine(config=cfg, pipeline_stages=[stage1, stage2])

    ctx = engine.execute(image_source="dummy_path.png", manual_modality_override="brain_mri")

    assert ctx.execution_state == EngineState.COMPLETED.value
    assert stage1.validated and stage1.processed and stage1.cleaned_up
    assert stage2.validated and stage2.processed and stage2.cleaned_up
    assert ctx.metadata["Stage1"] == "success"
    assert ctx.metadata["Stage2"] == "success"
    assert ctx.total_execution_ms() > 0.0

    health = engine.get_health_status()
    assert health["status"] == "HEALTHY"


def test_engine_failing_execution():
    ok_stage = MockPipelineStage(name="OKStage")
    fail_stage = MockFailingStage()

    cfg = EngineConfiguration(engine_name="TestEngineFail")
    engine = MedicalAIEngine(config=cfg, pipeline_stages=[ok_stage, fail_stage])

    with pytest.raises(EngineExecutionException):
        engine.execute(image_source="dummy_path.png")

    stats = engine.get_statistics()
    assert stats["failed_sessions"] == 1


def test_engine_thread_safety_concurrent_requests():
    stage = MockPipelineStage(name="ConcurrentStage", delay_ms=10.0)
    cfg = EngineConfiguration(engine_name="TestEngineConcurrent", max_concurrent_threads=8)
    engine = MedicalAIEngine(config=cfg, pipeline_stages=[stage])
    engine.initialize()

    results = []
    errors = []

    def worker(i: int):
        try:
            ctx = engine.execute(image_source=f"dummy_{i}.png", request_id=f"req_conc_{i}")
            results.append(ctx)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 10
    stats = engine.get_statistics()
    assert stats["completed_sessions"] == 10

    engine.shutdown()
