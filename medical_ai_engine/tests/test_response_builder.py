"""
Module 14.2.5: Unified Response Builder & Clinical Report Aggregator
PyTest Test Suite

Tests:
  1. Immutable PredictionResult — frozen dataclass enforcement
  2. Successful response construction from complete EngineContext
  3. Clinical report aggregation — all fields populated
  4. Validation — required field detection
  5. Serialization — json / pretty_json / dict
  6. Error response construction via build_error_response()
  7. Missing field handling — partial EngineContext
  8. Concurrent response building — thread safety
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_builder import ResponseBuilder
from medical_ai_engine.response.response_schema import PredictionStatus, PredictionReliability
from medical_ai_engine.response.response_exceptions import (
    ResponseValidationException,
    ResponseSerializationException,
)
from medical_ai_engine.response.clinical_report_aggregator import ClinicalReportAggregator
from medical_ai_engine.response.response_validator import ResponseValidator
from medical_ai_engine.response.response_serializer import ResponseSerializer
from medical_ai_engine.response.response_configuration import ResponseConfiguration


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ctx(request_id: str = None, **overrides) -> EngineContext:
    """Build a fully-populated test EngineContext."""
    rid = request_id or str(uuid.uuid4())
    sid = str(uuid.uuid4())
    ctx = EngineContext(request_id=rid, session_id=sid)

    ctx.modality_prediction = {
        "detected_modality": "chest_xray",
        "modality_confidence": 0.97,
        "routing_mode": "automatic",
        "routing_time_ms": 5.3,
        "low_confidence_warning": False,
    }
    ctx.disease_prediction = {
        "predicted_class": "pneumonia",
        "confidence": 0.9231,
        "top_k_predictions": [
            {"class": "pneumonia", "confidence": 0.9231},
            {"class": "normal", "confidence": 0.0521},
            {"class": "pleural_effusion", "confidence": 0.0248},
        ],
        "model_name": "ChestXrayClassifier",
        "model_version": "1.2.0",
        "inference_backend": "pytorch",
        "inference_time_ms": 42.1,
    }
    ctx.confidence = {
        "score": 0.9231,
        "reliability": "high",
        "requires_manual_review": False,
        "severity_grade": None,
    }
    ctx.clinical_report = {
        "report_type": "Chest X-Ray Diagnostic Report",
        "primary_diagnosis": "pneumonia",
        "reliability": "high",
        "requires_manual_review": False,
        "differential_diagnoses": [],
        "clinical_note": "AI-assisted analysis.",
    }
    ctx.metadata["selected_adapter"] = "ChestXrayAdapter"
    ctx.performance_metrics = {
        "validation_ms": 1.2,
        "routing_ms": 5.3,
        "inference_ms": 42.1,
        "total_pipeline_ms": 60.5,
    }

    for key, val in overrides.items():
        object.__setattr__(ctx, key, val)

    return ctx


# ── Test 1: Immutable PredictionResult ────────────────────────────────────────

class TestPredictionResultImmutability:

    def test_frozen_dataclass_raises_on_mutation(self):
        """PredictionResult must be immutable (frozen=True)."""
        result = PredictionResult(
            request_id="r-001",
            session_id="s-001",
            status=PredictionStatus.SUCCESS,
            modality="chest_xray",
            predicted_disease="pneumonia",
            confidence=0.92,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.confidence = 0.5  # Must fail

    def test_status_is_enum(self):
        result = PredictionResult(
            request_id="r-002", session_id="s-002",
            status=PredictionStatus.SUCCESS,
        )
        assert isinstance(result.status, PredictionStatus)

    def test_is_success_property(self):
        r = PredictionResult(
            request_id="r-003", session_id="s-003",
            status=PredictionStatus.SUCCESS,
        )
        assert r.is_success is True
        assert r.is_failed is False

    def test_is_failed_property(self):
        r = PredictionResult(
            request_id="r-004", session_id="s-004",
            status=PredictionStatus.FAILED,
        )
        assert r.is_failed is True
        assert r.is_success is False


# ── Test 2: Successful Response Construction ──────────────────────────────────

class TestResponseBuilderSuccess:

    def test_build_from_complete_context(self):
        ctx = _make_ctx()
        builder = ResponseBuilder()
        result = builder.build(ctx)

        assert isinstance(result, PredictionResult)
        assert result.request_id == ctx.request_id
        assert result.session_id == ctx.session_id
        assert result.modality == "chest_xray"
        assert result.predicted_disease == "pneumonia"
        assert abs(result.confidence - 0.9231) < 1e-5
        assert result.prediction_reliability == PredictionReliability.HIGH
        assert result.requires_manual_review is False
        assert result.status == PredictionStatus.SUCCESS

    def test_top_k_predictions_populated(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert len(result.top_predictions) == 3
        assert result.top_predictions[0]["class"] == "pneumonia"

    def test_performance_metrics_carried(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert "total_pipeline_ms" in result.performance_metrics
        assert result.performance_metrics["total_pipeline_ms"] == pytest.approx(60.5)

    def test_model_metadata_extracted(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert result.model_metadata.get("model_name") == "ChestXrayClassifier"
        assert result.model_metadata.get("inference_backend") == "pytorch"

    def test_routing_decision_extracted(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert result.routing_decision.get("detected_modality") == "chest_xray"
        assert result.routing_decision.get("routing_mode") == "automatic"


# ── Test 3: Clinical Report Aggregation ───────────────────────────────────────

class TestClinicalReportAggregation:

    def test_clinical_report_keys_present(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        report = result.clinical_report
        for key in ("primary_diagnosis", "reliability", "requires_manual_review", "recommendations"):
            assert key in report, f"Expected '{key}' in clinical_report"

    def test_recommendations_list_not_empty(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert isinstance(result.clinical_report["recommendations"], list)
        assert len(result.clinical_report["recommendations"]) > 0

    def test_clinical_note_contains_diagnosis(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        assert "pneumonia" in result.clinical_report.get("clinical_note", "").lower()

    def test_differential_diagnoses_populated(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        # Aggregator populates from top_k_predictions in disease_prediction
        assert isinstance(result.clinical_report.get("differential_diagnoses", []), list)

    def test_aggregator_directly(self):
        agg = ClinicalReportAggregator()
        disease_pred = {
            "predicted_class": "melanoma",
            "confidence": 0.88,
            "top_k_predictions": [{"class": "melanoma", "confidence": 0.88}],
        }
        confidence_data = {"reliability": "high", "requires_manual_review": False}
        report = agg.aggregate("skin", disease_pred, confidence_data)
        assert report["primary_diagnosis"] == "melanoma"
        assert report["reliability"] == "high"


# ── Test 4: Validation ────────────────────────────────────────────────────────

class TestResponseValidation:

    def test_valid_result_passes_validation(self):
        ctx = _make_ctx()
        result = ResponseBuilder().build(ctx)
        validator = ResponseValidator()
        errors = validator.get_validation_errors(result)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_confidence_out_of_range_is_caught(self):
        result = PredictionResult(
            request_id="r-x01", session_id="s-x01",
            status=PredictionStatus.FAILED,
            confidence=1.5,  # invalid
        )
        validator = ResponseValidator(ResponseConfiguration(strict_validation=False))
        errors = validator.get_validation_errors(result)
        assert any("confidence" in e for e in errors)

    def test_empty_request_id_is_caught(self):
        result = PredictionResult(
            request_id="", session_id="s-x02",
            status=PredictionStatus.FAILED,
        )
        validator = ResponseValidator(ResponseConfiguration(strict_validation=False))
        errors = validator.get_validation_errors(result)
        assert any("request_id" in e for e in errors)

    def test_strict_validation_raises_on_error(self):
        result = PredictionResult(
            request_id="", session_id="s-x03",
            status=PredictionStatus.FAILED,
        )
        validator = ResponseValidator(ResponseConfiguration(strict_validation=True))
        with pytest.raises(ResponseValidationException):
            validator.validate(result)


# ── Test 5: Serialization ─────────────────────────────────────────────────────

class TestResponseSerialization:

    def _sample_result(self) -> PredictionResult:
        ctx = _make_ctx()
        return ResponseBuilder().build(ctx)

    def test_to_json_produces_valid_json(self):
        result = self._sample_result()
        s = ResponseSerializer()
        json_str = s.serialize_to_json(result)
        parsed = json.loads(json_str)
        assert parsed["modality"] == "chest_xray"
        assert parsed["predicted_disease"] == "pneumonia"

    def test_to_pretty_json_has_indentation(self):
        result = self._sample_result()
        s = ResponseSerializer()
        pretty = s.serialize_to_pretty_json(result)
        assert "\n" in pretty

    def test_to_dict_returns_dict(self):
        result = self._sample_result()
        s = ResponseSerializer()
        d = s.serialize_to_dict(result)
        assert isinstance(d, dict)
        assert d["status"] == PredictionStatus.SUCCESS.value

    def test_unsupported_format_raises(self):
        result = self._sample_result()
        s = ResponseSerializer()
        with pytest.raises(ResponseSerializationException):
            s.serialize(result, fmt="fhir_r4")

    def test_enum_serialized_as_string(self):
        result = self._sample_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] == "SUCCESS"
        assert isinstance(parsed["prediction_reliability"], str)


# ── Test 6: Error Response Construction ───────────────────────────────────────

class TestErrorResponseConstruction:

    def test_build_error_response_returns_failed_status(self):
        ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
        builder = ResponseBuilder()
        result = builder.build_error_response(ctx, "Simulated pipeline crash")

        assert result.status == PredictionStatus.FAILED
        assert result.predicted_disease == "unknown"
        assert result.confidence == 0.0
        assert result.requires_manual_review is True

    def test_error_result_contains_error_message_in_report(self):
        ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
        builder = ResponseBuilder()
        result = builder.build_error_response(ctx, "Adapter not registered")
        assert "Adapter not registered" in result.clinical_report.get("clinical_note", "")

    def test_error_result_is_frozen(self):
        ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
        result = ResponseBuilder().build_error_response(ctx, "failure")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.status = PredictionStatus.SUCCESS


# ── Test 7: Missing Field Handling ────────────────────────────────────────────

class TestMissingFieldHandling:

    def test_partial_context_uses_fallbacks(self):
        """ResponseBuilder must not crash on a partial EngineContext."""
        ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
        # No disease_prediction, modality_prediction, confidence set
        builder = ResponseBuilder(config=ResponseConfiguration(strict_validation=False))
        result = builder.build_error_response(ctx, "Missing adapter output")
        assert result.status == PredictionStatus.FAILED
        assert result.modality == "unknown"

    def test_no_performance_metrics_returns_empty_dict(self):
        ctx = EngineContext(request_id=str(uuid.uuid4()), session_id=str(uuid.uuid4()))
        result = ResponseBuilder(config=ResponseConfiguration(strict_validation=False)).build_error_response(
            ctx, "Empty"
        )
        assert isinstance(result.performance_metrics, dict)


# ── Test 8: Concurrent Response Building ─────────────────────────────────────

class TestConcurrentResponseBuilding:

    def test_concurrent_builds_are_independent(self):
        """Concurrent builds from different EngineContext objects must not collide."""
        results: List[PredictionResult] = []
        errors: List[Exception] = []
        builder = ResponseBuilder()

        def build_one(thread_id: int) -> None:
            try:
                ctx = _make_ctx()
                ctx.disease_prediction = dict(ctx.disease_prediction, predicted_class=f"disease_{thread_id}")
                result = builder.build(ctx)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=build_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent build errors: {errors}"
        assert len(results) == 20
        # Each result should have a unique request_id
        request_ids = {r.request_id for r in results}
        assert len(request_ids) == 20
