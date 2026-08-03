"""
Module 14.2.5: Unified Response Builder & Clinical Report Aggregator
Standalone Verification Runner

Usage:
    python run_response_builder_tests.py
"""

from __future__ import annotations

import json
import sys
import time
import threading
import traceback
import uuid
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def ok(msg: str) -> None:
    print(f"    {GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"    {RED}[FAIL]{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 60}{RESET}")


# ── Build a full test EngineContext ───────────────────────────────────────────

def _make_ctx(disease: str = "pneumonia", confidence: float = 0.9231):
    from medical_ai_engine.core.engine_context import EngineContext
    ctx = EngineContext(
        request_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
    )
    ctx.modality_prediction = {
        "detected_modality": "chest_xray",
        "modality_confidence": 0.97,
        "routing_mode": "automatic",
        "routing_time_ms": 5.3,
    }
    ctx.disease_prediction = {
        "predicted_class": disease,
        "confidence": confidence,
        "top_k_predictions": [
            {"class": disease, "confidence": confidence},
            {"class": "normal", "confidence": 0.042},
        ],
        "model_name": "ChestXrayClassifier",
        "model_version": "1.2.0",
        "inference_backend": "pytorch",
        "inference_time_ms": 42.1,
    }
    ctx.confidence = {
        "score": confidence,
        "reliability": "high",
        "requires_manual_review": False,
        "severity_grade": None,
    }
    ctx.clinical_report = {
        "report_type": "Chest X-Ray Diagnostic Report",
        "primary_diagnosis": disease,
        "reliability": "high",
        "requires_manual_review": False,
    }
    ctx.metadata["selected_adapter"] = "ChestXrayAdapter"
    ctx.performance_metrics = {
        "validation_ms": 1.2,
        "routing_ms": 5.3,
        "inference_ms": 42.1,
        "total_pipeline_ms": 60.5,
    }
    return ctx


# ── Tests ─────────────────────────────────────────────────────────────────────

passed = 0
failed_count = 0


def assert_true(cond: bool, msg: str) -> None:
    global passed, failed_count
    if cond:
        passed += 1
        ok(msg)
    else:
        failed_count += 1
        fail(f"FAILED: {msg}")


def run_test1() -> None:
    section("[1/8] PredictionResult Immutability")
    from medical_ai_engine.response.prediction_result import PredictionResult
    from medical_ai_engine.response.response_schema import PredictionStatus

    r = PredictionResult(request_id="r-001", session_id="s-001",
                         status=PredictionStatus.SUCCESS, modality="chest_xray",
                         predicted_disease="pneumonia", confidence=0.92)
    try:
        r.confidence = 0.5
        assert_true(False, "Mutation should have raised")
    except (FrozenInstanceError, AttributeError):
        assert_true(True, "Mutation raises FrozenInstanceError or AttributeError")

    assert_true(r.is_success, "is_success is True for SUCCESS status")
    assert_true(not r.is_failed, "is_failed is False for SUCCESS status")

    r2 = PredictionResult(request_id="r-002", session_id="s-002", status=PredictionStatus.FAILED)
    assert_true(r2.is_failed, "is_failed is True for FAILED status")
    assert_true(not r2.is_success, "is_success is False for FAILED status")


def run_test2() -> None:
    section("[2/8] Successful Response Construction")
    from medical_ai_engine.response.response_builder import ResponseBuilder
    from medical_ai_engine.response.response_schema import PredictionStatus, PredictionReliability

    ctx = _make_ctx()
    result = ResponseBuilder().build(ctx)

    assert_true(result.modality == "chest_xray", "modality == 'chest_xray'")
    assert_true(result.predicted_disease == "pneumonia", "predicted_disease == 'pneumonia'")
    assert_true(abs(result.confidence - 0.9231) < 1e-5, f"confidence ≈ 0.9231 (got {result.confidence:.6f})")
    assert_true(result.prediction_reliability == PredictionReliability.HIGH, "reliability = HIGH")
    assert_true(result.requires_manual_review is False, "requires_manual_review = False")
    assert_true(result.status == PredictionStatus.SUCCESS, "status = SUCCESS")
    assert_true(result.model_metadata.get("model_name") == "ChestXrayClassifier", "model_name extracted")
    assert_true(result.routing_decision.get("detected_modality") == "chest_xray", "routing_decision populated")
    assert_true("total_pipeline_ms" in result.performance_metrics, "performance_metrics carried")
    print(f"\n    {YELLOW}→ PredictionResult:{RESET}")
    print(f"      {result!r}")


def run_test3() -> None:
    section("[3/8] Clinical Report Aggregation")
    from medical_ai_engine.response.clinical_report_aggregator import ClinicalReportAggregator

    agg = ClinicalReportAggregator()
    disease_pred = {
        "predicted_class": "melanoma",
        "confidence": 0.88,
        "top_k_predictions": [
            {"class": "melanoma", "confidence": 0.88},
            {"class": "benign", "confidence": 0.12},
        ],
    }
    confidence_data = {"reliability": "high", "requires_manual_review": False}
    report = agg.aggregate("skin", disease_pred, confidence_data)

    assert_true(report["primary_diagnosis"] == "melanoma", "primary_diagnosis = 'melanoma'")
    assert_true(report["reliability"] == "high", "reliability = 'high'")
    assert_true(len(report["differential_diagnoses"]) == 2, "2 differential diagnoses")
    assert_true("melanoma" in report.get("clinical_note", ""), "clinical_note contains 'melanoma'")
    assert_true(isinstance(report["recommendations"], list), "recommendations is a list")
    assert_true(len(report["recommendations"]) > 0, "recommendations non-empty")
    print(f"\n    {YELLOW}→ Report type:{RESET} {report['report_type']}")


def run_test4() -> None:
    section("[4/8] Validation")
    from medical_ai_engine.response.response_validator import ResponseValidator
    from medical_ai_engine.response.response_configuration import ResponseConfiguration
    from medical_ai_engine.response.prediction_result import PredictionResult
    from medical_ai_engine.response.response_schema import PredictionStatus
    from medical_ai_engine.response.response_exceptions import ResponseValidationException

    # Valid result should have no errors
    ctx = _make_ctx()
    from medical_ai_engine.response.response_builder import ResponseBuilder
    result = ResponseBuilder().build(ctx)
    validator = ResponseValidator()
    errors = validator.get_validation_errors(result)
    assert_true(errors == [], f"Valid result has no errors (got: {errors})")

    # confidence out of range
    bad = PredictionResult(request_id="r-b01", session_id="s-b01", confidence=1.5,
                           status=PredictionStatus.FAILED)
    v2 = ResponseValidator(ResponseConfiguration(strict_validation=False))
    errs = v2.get_validation_errors(bad)
    assert_true(any("confidence" in e for e in errs), "confidence=1.5 caught by validator")

    # strict mode raises
    try:
        v3 = ResponseValidator(ResponseConfiguration(strict_validation=True))
        v3.validate(bad)
        assert_true(False, "strict_validation should have raised")
    except ResponseValidationException:
        assert_true(True, "strict_validation raises ResponseValidationException")


def run_test5() -> None:
    section("[5/8] Serialization")
    from medical_ai_engine.response.response_builder import ResponseBuilder
    from medical_ai_engine.response.response_serializer import ResponseSerializer
    from medical_ai_engine.response.response_schema import PredictionStatus
    from medical_ai_engine.response.response_exceptions import ResponseSerializationException

    ctx = _make_ctx()
    result = ResponseBuilder().build(ctx)
    s = ResponseSerializer()

    json_str = s.serialize_to_json(result)
    parsed = json.loads(json_str)
    assert_true(parsed["modality"] == "chest_xray", "JSON → modality == 'chest_xray'")
    assert_true(parsed["status"] == "SUCCESS", "JSON → status == 'SUCCESS'")

    pretty = s.serialize_to_pretty_json(result)
    assert_true("\n" in pretty, "pretty_json contains newlines")

    d = s.serialize_to_dict(result)
    assert_true(isinstance(d, dict), "dict output is dict")
    assert_true(d["status"] == "SUCCESS", "dict → status == 'SUCCESS'")

    try:
        s.serialize(result, fmt="fhir_r4")
        assert_true(False, "fhir_r4 should have raised")
    except ResponseSerializationException:
        assert_true(True, "Unsupported 'fhir_r4' format raises ResponseSerializationException")


def run_test6() -> None:
    section("[6/8] Error Response Construction")
    from medical_ai_engine.core.engine_context import EngineContext
    from medical_ai_engine.response.response_builder import ResponseBuilder
    from medical_ai_engine.response.response_schema import PredictionStatus

    ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
    builder = ResponseBuilder()
    result = builder.build_error_response(ctx, "Adapter not registered for: skin")

    assert_true(result.status == PredictionStatus.FAILED, "Error response has FAILED status")
    assert_true(result.predicted_disease == "unknown", "predicted_disease = 'unknown'")
    assert_true(result.confidence == 0.0, "confidence = 0.0")
    assert_true(result.requires_manual_review is True, "requires_manual_review = True")
    assert_true(
        "Adapter not registered for: skin" in result.clinical_report.get("clinical_note", ""),
        "error message in clinical_note",
    )
    print(f"\n    {YELLOW}→ Error status:{RESET} {result.status.value}")


def run_test7() -> None:
    section("[7/8] Missing Field Handling")
    from medical_ai_engine.core.engine_context import EngineContext
    from medical_ai_engine.response.response_builder import ResponseBuilder
    from medical_ai_engine.response.response_configuration import ResponseConfiguration

    ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
    builder = ResponseBuilder(config=ResponseConfiguration(strict_validation=False))
    result = builder.build_error_response(ctx, "Missing adapter output")
    assert_true(result.modality == "unknown", "modality fallback = 'unknown'")
    assert_true(isinstance(result.performance_metrics, dict), "performance_metrics fallback is dict")
    assert_true(isinstance(result.warnings, list), "warnings fallback is list")


def run_test8() -> None:
    section("[8/8] Concurrent Response Building")
    from medical_ai_engine.response.response_builder import ResponseBuilder
    from medical_ai_engine.response.prediction_result import PredictionResult

    results: List[PredictionResult] = []
    errors: List[Exception] = []
    builder = ResponseBuilder()

    def build_one(thread_id: int) -> None:
        try:
            ctx = _make_ctx(disease=f"disease_{thread_id}", confidence=0.80 + thread_id * 0.001)
            result = builder.build(ctx)
            results.append(result)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=build_one, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert_true(errors == [], f"No errors in 20-thread concurrent build (errors: {errors})")
    assert_true(len(results) == 20, f"20/20 results built (got {len(results)})")
    request_ids = {r.request_id for r in results}
    assert_true(len(request_ids) == 20, "All 20 request_ids are unique")
    print(f"\n    {YELLOW}→ All 20 concurrent builds succeeded.{RESET}")


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print("  Module 14.2.5: Unified Response Builder -- Verification Runner")
    print(f"{'=' * 60}{RESET}\n")

    start = time.perf_counter()

    for runner in [
        run_test1, run_test2, run_test3, run_test4,
        run_test5, run_test6, run_test7, run_test8,
    ]:
        try:
            runner()
        except Exception:
            print(f"  {RED}FATAL: {runner.__name__} raised uncaught exception:{RESET}")
            traceback.print_exc()
            global failed_count
            failed_count += 1

    elapsed = (time.perf_counter() - start) * 1000

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {GREEN}{passed} passed{RESET}  {RED}{failed_count} failed{RESET}  [{elapsed:.1f}ms]")
    print(f"{'=' * 60}\n")

    if failed_count == 0:
        print(f"{GREEN}{BOLD}  [SUCCESS] All Module 14.2.5 Response Builder tests PASSED!{RESET}\n")
    else:
        print(f"{RED}{BOLD}  [FAILURE] {failed_count} test(s) FAILED.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
