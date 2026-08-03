"""
Module 14.2.6: End-to-End System Integration
Standalone Verification Runner

Usage:
    python run_system_integration_tests.py
"""

from __future__ import annotations

import json
import sys
import time
import threading
import traceback
import uuid
from typing import Any, Dict, List
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

passed = 0
failed_count = 0


def ok(msg: str) -> None:
    print(f"    {GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"    {RED}[FAIL]{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")


def assert_true(cond: bool, msg: str) -> None:
    global passed, failed_count
    if cond:
        passed += 1
        ok(msg)
    else:
        failed_count += 1
        fail(f"FAILED: {msg}")


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (128, 128), color=(120, 140, 160))


# ── Test 1: System Integration Initialization ─────────────────────────────────

def run_test1() -> EngineIntegrator:
    section("[1/5] System Integration Initialization & Health Check")
    from medical_ai_engine.integration.engine_integrator import EngineIntegrator
    from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
    from medical_ai_engine.core.engine_state import EngineState

    cfg = IntegrationConfiguration.test_config()
    integrator = EngineIntegrator(config=cfg)

    assert_true(integrator.engine.state == EngineState.INITIALIZED, "MedicalAIEngine core is INITIALIZED")

    health = integrator.health_check()
    assert_true(health["status"] == "healthy", f"System health == 'healthy' (got {health['status']})")
    assert_true(health["registered_adapters_count"] >= 7, f"Registered adapters >= 7 (got {health['registered_adapters_count']})")
    print(f"\n    {YELLOW}Registered modalities:{RESET} {health['registered_modalities']}")
    return integrator


# ── Test 2: All 7 Modality End-to-End Workflows ───────────────────────────────

def run_test2(integrator: EngineIntegrator) -> None:
    section("[2/5] End-to-End Workflows for All 7 Supported Disease Modalities")
    from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
    from medical_ai_engine.response.prediction_result import PredictionResult

    modalities = [
        "brain_mri", "chest_xray", "brain_ct", "histopathology",
        "retina", "skin", "ultrasound",
    ]

    executor = WorkflowExecutor(integrator=integrator)
    img = _make_dummy_image()

    for mod in modalities:
        res, serialized, ctx = executor.execute(
            image_source=img,
            manual_modality_override=mod,
            format="dict",
            expected_modality=mod,
        )

        assert_true(isinstance(res, PredictionResult), f"[{mod}] PredictionResult created")
        assert_true(res.modality == mod, f"[{mod}] result.modality == '{mod}'")
        assert_true(res.predicted_disease != "unknown", f"[{mod}] disease != 'unknown'")
        assert_true(res.confidence > 0.0, f"[{mod}] confidence > 0.0 ({res.confidence:.4f})")
        assert_true(isinstance(serialized, dict), f"[{mod}] Serialized output is dict")
        assert_true("primary_diagnosis" in res.clinical_report, f"[{mod}] Clinical report populated")


# ── Test 3: Failure Recovery & Edge Case Tests ───────────────────────────────

def run_test3() -> None:
    section("[3/5] Failure Recovery & Resiliency Tests")
    from medical_ai_engine.integration.engine_integrator import EngineIntegrator
    from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
    from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
    from medical_ai_engine.response.response_schema import PredictionStatus

    img = _make_dummy_image()

    # 1. Unsupported Modality
    cfg = IntegrationConfiguration(auto_register_adapters=False, catch_pipeline_exceptions=True, enable_strict_validation=False)
    integrator = EngineIntegrator(config=cfg)
    executor = WorkflowExecutor(integrator=integrator)

    res, serialized, ctx = executor.execute(image_source=img, manual_modality_override="quantum_pet_scan")
    assert_true(res.status in (PredictionStatus.FAILED, PredictionStatus.COMPLETED_WITH_WARNINGS), "Unsupported modality returns non-crash status")
    assert_true(res.requires_manual_review is True, "Requires manual review flag set")

    # 2. Missing Adapter
    res2, _, _ = executor.execute(image_source=img, manual_modality_override="nonexistent_modality")
    assert_true(res2.status in (PredictionStatus.FAILED, PredictionStatus.COMPLETED_WITH_WARNINGS), "Missing adapter recovers gracefully without crash")


# ── Test 4: Performance Metrics & Latency Breakdown ─────────────────────────

def run_test4(integrator: EngineIntegrator) -> None:
    section("[4/5] Integration Metrics & Latency Breakdown")
    from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
    from medical_ai_engine.integration.integration_metrics import IntegrationMetrics

    metrics = IntegrationMetrics()
    executor = WorkflowExecutor(integrator=integrator, metrics=metrics)
    img = _make_dummy_image()

    for mod in ["brain_mri", "chest_xray", "skin"]:
        executor.execute(image_source=img, manual_modality_override=mod)

    summary = metrics.get_summary()
    assert_true(summary["total_workflows"] == 3, "Recorded 3 total workflows")
    assert_true(summary["successful_workflows"] == 3, "Recorded 3 successful workflows")
    assert_true(summary["success_rate_pct"] == 100.0, "100% success rate")
    assert_true(summary["avg_total_execution_ms"] >= 0.0, f"Avg latency: {summary['avg_total_execution_ms']:.2f}ms")
    print(f"\n    {YELLOW}Summary metrics dict:{RESET}")
    print(f"      {summary}")


# ── Test 5: Concurrent 20-thread Workflow Execution ───────────────────────────

def run_test5(integrator: EngineIntegrator) -> None:
    section("[5/5] Concurrent 20-thread Workflow Execution")
    from medical_ai_engine.integration.workflow_executor import WorkflowExecutor

    executor = WorkflowExecutor(integrator=integrator)
    results: List[Any] = []
    errors: List[Exception] = []

    modalities = ["brain_mri", "chest_xray", "brain_ct", "histopathology", "retina", "skin", "ultrasound"]

    def worker(thread_id: int):
        try:
            img = _make_dummy_image()
            target_mod = modalities[thread_id % len(modalities)]
            res, _, _ = executor.execute(image_source=img, manual_modality_override=target_mod)
            results.append(res)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert_true(errors == [], f"No errors during 20-thread execution (errors: {errors})")
    assert_true(len(results) == 20, f"20/20 workflows completed (got {len(results)})")
    req_ids = {r.request_id for r in results}
    assert_true(len(req_ids) == 20, "All 20 request_ids are unique")


# ── Main Runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'=' * 65}")
    print("  Module 14.2.6: End-to-End System Integration -- Verification")
    print(f"{'=' * 65}{RESET}\n")

    start = time.perf_counter()

    try:
        integrator = run_test1()
        run_test2(integrator)
        run_test3()
        run_test4(integrator)
        run_test5(integrator)
    except Exception:
        print(f"\n  {RED}FATAL: Uncaught exception in integration runner:{RESET}")
        traceback.print_exc()
        global failed_count
        failed_count += 1

    elapsed = (time.perf_counter() - start) * 1000.0

    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {GREEN}{passed} passed{RESET}  {RED}{failed_count} failed{RESET}  [{elapsed:.1f}ms]")
    print(f"{'=' * 65}\n")

    if failed_count == 0:
        print(f"{GREEN}{BOLD}  [SUCCESS] All Module 14.2.6 End-to-End Integration tests PASSED!{RESET}\n")
    else:
        print(f"{RED}{BOLD}  [FAILURE] {failed_count} test(s) FAILED.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
