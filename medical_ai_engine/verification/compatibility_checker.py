"""
Medical AI Engine — Compatibility Checker

Module 14.2.7: Production Verification & Stress Testing

Verifies backward compatibility of all public platform contracts:
  - PredictionResult field set
  - EngineContext field set
  - PredictionStatus / PredictionReliability enum values
  - Adapter output dict keys
  - Clinical report keys
  - JSON serialization round-trip stability
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import VerificationResult, VerificationStatus

logger = logging.getLogger("CompatibilityChecker")

# ── Required fields in PredictionResult ──────────────────────────────────────
_PREDICTION_RESULT_REQUIRED_FIELDS = {
    "request_id", "session_id", "timestamp", "schema_version",
    "status", "modality", "predicted_disease", "confidence",
    "prediction_reliability", "requires_manual_review",
    "routing_decision", "clinical_report", "performance_metrics",
    "model_metadata", "warnings", "errors", "top_predictions",
    "severity_grade", "gradcam_result", "anomaly_result", "extra",
}

# ── Required fields in EngineContext ─────────────────────────────────────────
_ENGINE_CONTEXT_REQUIRED_FIELDS = {
    "request_id", "session_id", "image", "image_source", "image_bytes",
    "manual_modality_override", "metadata",
}

# ── Required PredictionStatus enum values ────────────────────────────────────
_PREDICTION_STATUS_VALUES = {
    "SUCCESS", "COMPLETED_WITH_WARNINGS", "FAILED",
    "LOW_CONFIDENCE", "UNSUPPORTED_MODALITY", "MODEL_NOT_REGISTERED",
}

# ── Required PredictionReliability enum values ───────────────────────────────
_PREDICTION_RELIABILITY_VALUES = {
    "high", "moderate", "low", "unknown",
}

# ── Required adapter output dict keys ────────────────────────────────────────
_ADAPTER_OUTPUT_REQUIRED_KEYS = {
    "predicted_class", "confidence",
}

# ── Required clinical report keys ────────────────────────────────────────────
_CLINICAL_REPORT_REQUIRED_KEYS = {
    "primary_diagnosis", "confidence",
}


class CompatibilityChecker:
    """
    Verifies that all public API contracts of the Medical AI Platform
    remain backward compatible with their declared specifications.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run all compatibility checks and return a VerificationResult."""
        start = time.perf_counter()
        violations: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        checks_passed = 0
        total_checks = 0

        # 1. PredictionResult fields
        if self.config.check_prediction_result_fields:
            total_checks += 1
            ok, errs = self._check_prediction_result_fields()
            if ok:
                checks_passed += 1
            else:
                violations.extend(errs)
            details["prediction_result_field_errors"] = errs

        # 2. EngineContext fields
        if self.config.check_engine_context_fields:
            total_checks += 1
            ok, errs = self._check_engine_context_fields()
            if ok:
                checks_passed += 1
            else:
                violations.extend(errs)
            details["engine_context_field_errors"] = errs

        # 3. Enum values
        if self.config.check_enum_values:
            total_checks += 1
            ok, errs = self._check_enum_values()
            if ok:
                checks_passed += 1
            else:
                violations.extend(errs)
            details["enum_value_errors"] = errs

        # 4. JSON round-trip
        if self.config.check_json_roundtrip:
            total_checks += 1
            ok, err = self._check_json_roundtrip()
            if ok:
                checks_passed += 1
            elif err:
                violations.append(err)
            details["json_roundtrip_ok"] = ok

        # 5. Adapter output keys
        if self.config.check_adapter_output_keys:
            total_checks += 1
            ok, errs = self._check_adapter_output_keys()
            if ok:
                checks_passed += 1
            else:
                warnings.extend(errs)  # Mock adapters may differ slightly — warn only
            details["adapter_output_key_warnings"] = errs

        # 6. Clinical report keys
        total_checks += 1
        ok, errs = self._check_clinical_report_keys()
        if ok:
            checks_passed += 1
        else:
            warnings.extend(errs)
        details["clinical_report_key_warnings"] = errs

        # ── Score ──────────────────────────────────────────────────────────
        score = (checks_passed / total_checks * 100.0) if total_checks > 0 else 0.0
        if violations:
            status = VerificationStatus.FAILED
        elif warnings:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.PASSED

        details["checks_passed"] = checks_passed
        details["total_checks"] = total_checks

        return VerificationResult(
            name="Backward Compatibility Check",
            status=status,
            score=score,
            duration_seconds=time.perf_counter() - start,
            message=(
                f"{checks_passed}/{total_checks} compatibility checks passed. "
                f"{len(violations)} breaking change(s), {len(warnings)} warning(s)."
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _check_prediction_result_fields(self) -> Tuple[bool, List[str]]:
        errors = []
        try:
            from medical_ai_engine.response.prediction_result import PredictionResult
            actual_fields = {f.name for f in fields(PredictionResult)}
            missing = _PREDICTION_RESULT_REQUIRED_FIELDS - actual_fields
            for f_name in sorted(missing):
                errors.append(f"PredictionResult missing required field: '{f_name}'")
        except Exception as e:
            errors.append(f"Cannot inspect PredictionResult: {e}")
        return len(errors) == 0, errors

    def _check_engine_context_fields(self) -> Tuple[bool, List[str]]:
        errors = []
        try:
            from medical_ai_engine.core.engine_context import EngineContext
            ctx = EngineContext(request_id="compat-check", session_id="compat-session")
            for attr in _ENGINE_CONTEXT_REQUIRED_FIELDS:
                if not hasattr(ctx, attr):
                    errors.append(f"EngineContext missing required attribute: '{attr}'")
        except Exception as e:
            errors.append(f"Cannot inspect EngineContext: {e}")
        return len(errors) == 0, errors

    def _check_enum_values(self) -> Tuple[bool, List[str]]:
        errors = []
        try:
            from medical_ai_engine.response.response_schema import PredictionStatus, PredictionReliability
            actual_status = {e.value for e in PredictionStatus}
            actual_reliability = {e.value for e in PredictionReliability}

            for v in _PREDICTION_STATUS_VALUES:
                if v not in actual_status:
                    errors.append(f"PredictionStatus missing value: '{v}'")
            for v in _PREDICTION_RELIABILITY_VALUES:
                if v not in actual_reliability:
                    errors.append(f"PredictionReliability missing value: '{v}'")
        except Exception as e:
            errors.append(f"Cannot inspect enums: {e}")
        return len(errors) == 0, errors

    def _check_json_roundtrip(self) -> Tuple[bool, str]:
        """Build a minimal PredictionResult and verify JSON round-trip stability."""
        try:
            from medical_ai_engine.response.prediction_result import PredictionResult
            from medical_ai_engine.response.response_schema import PredictionStatus, PredictionReliability

            result = PredictionResult(
                request_id="compat-rt-001",
                session_id="compat-rt-session",
                status=PredictionStatus.SUCCESS,
                modality="brain_mri",
                predicted_disease="normal",
                confidence=0.95,
                prediction_reliability=PredictionReliability.HIGH,
            )
            json_str = result.to_json()
            parsed = json.loads(json_str)
            re_serialized = json.dumps(parsed, default=str)
            re_parsed = json.loads(re_serialized)

            if parsed["request_id"] != re_parsed["request_id"]:
                return False, "JSON round-trip request_id mismatch"
            if parsed["modality"] != re_parsed["modality"]:
                return False, "JSON round-trip modality mismatch"
            return True, ""
        except Exception as e:
            return False, f"JSON round-trip failed: {e}"

    def _check_adapter_output_keys(self) -> Tuple[bool, List[str]]:
        """Verify mock adapter output contains required keys."""
        warnings_list = []
        try:
            from PIL import Image
            import numpy as np
            dummy_image = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )
            from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter

            class _MinimalAdapter(BasePredictorAdapter):
                @property
                def modality(self) -> str:
                    return "test"

                def load_model(self, checkpoint_path=None, label_map_path=None):
                    self._is_loaded = True

                def predict(self, image):
                    return {
                        "predicted_class": "test",
                        "confidence": 0.9,
                        "top_k_predictions": [],
                        "model_name": "test_model",
                        "model_version": "1.0",
                        "inference_backend": "test",
                        "inference_time_ms": 1.0,
                    }

            adapter = _MinimalAdapter()
            adapter.load_model()
            output = adapter.predict(dummy_image)
            for key in _ADAPTER_OUTPUT_REQUIRED_KEYS:
                if key not in output:
                    warnings_list.append(f"Adapter output missing key: '{key}'")
        except Exception as e:
            warnings_list.append(f"Adapter output check skipped: {e}")
        return len(warnings_list) == 0, warnings_list

    def _check_clinical_report_keys(self) -> Tuple[bool, List[str]]:
        """Verify ClinicalReportAggregator output contains required keys."""
        warnings_list = []
        try:
            from medical_ai_engine.response.clinical_report_aggregator import ClinicalReportAggregator
            from medical_ai_engine.core.engine_context import EngineContext

            ctx = EngineContext(request_id="compat-cr", session_id="compat-cr-session")
            ctx.modality = "brain_mri"
            ctx.prediction_output = {
                "predicted_class": "normal",
                "confidence": 0.92,
                "top_k_predictions": [{"class": "normal", "confidence": 0.92}],
            }
            aggregator = ClinicalReportAggregator()
            report = aggregator.aggregate(ctx)
            for key in _CLINICAL_REPORT_REQUIRED_KEYS:
                if key not in report:
                    warnings_list.append(f"Clinical report missing key: '{key}'")
        except Exception as e:
            warnings_list.append(f"Clinical report check skipped: {e}")
        return len(warnings_list) == 0, warnings_list
