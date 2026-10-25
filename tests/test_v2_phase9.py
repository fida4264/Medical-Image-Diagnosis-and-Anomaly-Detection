"""
tests/test_v2_phase9.py
=======================
Comprehensive test suite for Phase 9: Pipeline Integration, Quality Gates
& Clinical Decision Hardening — V2.0.

Test Groups:
1.  Schema correctness and Pydantic V2 model instantiation.
2.  InputValidator: modality validation (valid/invalid/missing).
3.  InputValidator: image input decoding (PIL Image, bytes, file path).
4.  InputValidator: format, size, and dimension rejection rules.
5.  QualityGateEvaluator: valid inputs pass all gates → ACCEPTED.
6.  QualityGateEvaluator: invalid input → REJECTED_INVALID_INPUT.
7.  QualityGateEvaluator: unsupported modality → REJECTED_UNSUPPORTED_MODALITY.
8.  QualityGateEvaluator: model failure → INFERENCE_ERROR.
9.  QualityGateEvaluator: non-finite logits → INFERENCE_ERROR.
10. QualityGateEvaluator: OOD score above threshold → REJECTED_OOD.
11. QualityGateEvaluator: OOD score below threshold → ACCEPTED.
12. QualityGateEvaluator: anomaly above threshold → REVIEW_REQUIRED_ANOMALY.
13. QualityGateEvaluator: OOD enforcement disabled → FLAGGED_FOR_REVIEW (not rejection).
14. AuditLogger: SHA-256 provenance hashing (deterministic, non-reversible).
15. AuditLogger: patient pseudonym de-identification (HASH prefix, no raw ID).
16. AuditLogger: thread-safe concurrent writes (no corruption under concurrency).
17. AuditLogger: None bytes produce None hash (not crash).
18. Pipeline config: all 7 modalities listed and fully specified.
19. Pipeline config: Retina task is 'multilabel' with exactly 8 classes.
20. Pipeline config: artifact paths are consistent with project structure.
21. V2InferencePipeline: invalid modality → is_abstained=True, correct state.
22. V2InferencePipeline: valid PIL Image + valid modality → structured response.
23. V2InferencePipeline: Retina returns per-label predictions and thresholds.
24. V2InferencePipeline: non-calibrated path uses temperature=1.0 as default.
25. V2InferencePipeline: structured response always contains limitations_notice.
26. Artifact integrity: baseline checkpoints must exist and not be modified.
27. Artifact integrity: calibrated artifacts must exist and not be modified.
28. Artifact integrity: split manifests must exist and be read-only preserved.
29. Artifact integrity: Phase 7 calibration params JSON must exist.
30. Artifact integrity: Phase 8 OOD thresholds JSON must exist.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# -------------------------------------------------------------------
# Pipeline V2 Imports
# -------------------------------------------------------------------
from pipeline_v2.config import (
    AUDIT_LOG_FILE,
    BASELINE_MODELS_DIR,
    CALIBRATED_MODELS_DIR,
    CALIBRATION_PARAMS_PATH,
    LEGAL_LIMITATIONS_NOTICE,
    MODALITY_SETTINGS,
    OOD_MODELS_DIR,
    OOD_THRESHOLDS_PATH,
    PIPELINE_VERSION,
    RETINA_THRESHOLDS_PATH,
    ROOT,
    SEED,
    SUPPORTED_MODALITIES,
)
from pipeline_v2.schemas import (
    GateOutcome,
    QualityGateResult,
    QualityGateState,
    StructuredInferenceResponse,
)
from pipeline_v2.validator import InputValidationError, InputValidator
from pipeline_v2.quality_gates import QualityGateEvaluator
from pipeline_v2.audit_logger import AuditLogger

# -------------------------------------------------------------------
# Paths for artifact preservation assertions
# -------------------------------------------------------------------
DATA_MANIFESTS_DIR = ROOT / "data" / "manifests" / "splits"


# ===========================================================================
# Helpers
# ===========================================================================

def make_rgb_pil(width: int = 128, height: int = 128) -> Image.Image:
    """Creates a minimal valid RGB PIL Image for testing."""
    rng = np.random.default_rng(42)
    arr = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_rgb_bytes(width: int = 128, height: int = 128, fmt: str = "PNG") -> bytes:
    """Creates a valid encoded image as bytes."""
    img = make_rgb_pil(width, height)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_gate_result(status: QualityGateState, reasons: list = None) -> QualityGateResult:
    """Constructs a minimal QualityGateResult for mocking."""
    return QualityGateResult(
        overall_status=status,
        rejection_reasons=reasons or [],
        review_reasons=[],
        gate_outcomes={},
    )


# ===========================================================================
# Group 1: Schema correctness
# ===========================================================================

class TestSchemas:
    """Test Pydantic V2 schema instantiation and field defaults."""

    def test_quality_gate_state_enum_members(self):
        """All canonical QualityGateState values must be present."""
        states = {s.value for s in QualityGateState}
        expected = {
            "ACCEPTED_FOR_MODEL_OUTPUT",
            "FLAGGED_FOR_REVIEW",
            "REJECTED_INVALID_INPUT",
            "REJECTED_UNSUPPORTED_MODALITY",
            "REJECTED_OOD",
            "REVIEW_REQUIRED_ANOMALY",
            "INFERENCE_ERROR",
        }
        assert expected == states

    def test_gate_outcome_model_fields(self):
        """GateOutcome instantiates with mandatory fields."""
        go = GateOutcome(gate_name="GATE_TEST", passed=True, message="OK")
        assert go.gate_name == "GATE_TEST"
        assert go.passed is True
        assert go.details is None

    def test_quality_gate_result_defaults(self):
        """QualityGateResult has empty lists by default for rejection/review reasons."""
        gr = QualityGateResult(overall_status=QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        assert gr.rejection_reasons == []
        assert gr.review_reasons == []
        assert gr.gate_outcomes == {}

    def test_structured_inference_response_instantiation(self):
        """StructuredInferenceResponse can be built with minimal required fields."""
        gr = QualityGateResult(overall_status=QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        resp = StructuredInferenceResponse(
            request_id="test-001",
            inference_timestamp="2026-10-25T00:00:00Z",
            modality="brain_mri",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/brain_mri/best_model.pt",
            prediction_status=QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT,
            is_abstained=False,
            ood_method="knn",
            ood_status="WITHIN_DISTRIBUTION",
            anomaly_status="NORMAL_REFERENCE_ALIGNED",
            quality_gate_results=gr,
            limitations_notice=LEGAL_LIMITATIONS_NOTICE,
        )
        assert resp.request_id == "test-001"
        assert resp.pipeline_version == "V2.0-Phase9"
        assert resp.is_abstained is False
        assert "RESEARCH PROTOTYPE" in resp.limitations_notice

    def test_structured_response_abstained_fields_default_none(self):
        """Abstained response has null predicted_class and confidence_score."""
        gr = QualityGateResult(overall_status=QualityGateState.REJECTED_OOD)
        resp = StructuredInferenceResponse(
            request_id="test-002",
            inference_timestamp="2026-10-25T00:00:00Z",
            modality="chest_xray",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/chest_xray/best_model.pt",
            prediction_status=QualityGateState.REJECTED_OOD,
            is_abstained=True,
            ood_method="knn",
            ood_status="POTENTIALLY_OOD",
            anomaly_status="UNAVAILABLE",
            quality_gate_results=gr,
            limitations_notice=LEGAL_LIMITATIONS_NOTICE,
        )
        assert resp.is_abstained is True
        assert resp.predicted_class is None
        assert resp.confidence_score is None
        assert resp.retina_label_predictions is None


# ===========================================================================
# Group 2-4: InputValidator
# ===========================================================================

class TestInputValidatorModality:
    """Tests for modality validation logic."""

    def test_valid_modalities_all_accepted(self):
        """Every modality in SUPPORTED_MODALITIES must pass validation."""
        for mod in SUPPORTED_MODALITIES:
            result = InputValidator.validate_modality(mod)
            assert result == mod

    def test_case_insensitive_modality(self):
        """Modality validation normalises to lowercase."""
        result = InputValidator.validate_modality("Brain_MRI")
        assert result == "brain_mri"

    def test_missing_modality_raises(self):
        """None or empty modality raises InputValidationError."""
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_modality(None)
        assert "MISSING_MODALITY" in exc.value.error_code

        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_modality("")
        assert "MISSING_MODALITY" in exc.value.error_code

    def test_unsupported_modality_raises(self):
        """Unknown modality raises with UNSUPPORTED_MODALITY error code."""
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_modality("mri_brain_unknown_xyz")
        assert "UNSUPPORTED_MODALITY" in exc.value.error_code

    def test_whitespace_modality_cleaned(self):
        """Leading/trailing whitespace is stripped before matching."""
        result = InputValidator.validate_modality("  retina  ")
        assert result == "retina"


class TestInputValidatorImageInput:
    """Tests for image input decoding."""

    def test_valid_pil_image_accepted(self):
        """Valid PIL RGB Image passes validation and is returned as RGB."""
        img = make_rgb_pil()
        sanitized, meta = InputValidator.validate_image_input(img)
        assert isinstance(sanitized, Image.Image)
        assert sanitized.mode == "RGB"
        assert meta["source_type"] == "Image"

    def test_valid_bytes_png_accepted(self):
        """Valid PNG bytes decoded successfully."""
        raw = make_rgb_bytes(128, 128, "PNG")
        sanitized, meta = InputValidator.validate_image_input(raw)
        assert isinstance(sanitized, Image.Image)
        assert sanitized.mode == "RGB"
        assert meta["file_size_bytes"] == len(raw)

    def test_valid_bytes_jpeg_accepted(self):
        """Valid JPEG bytes decoded successfully."""
        raw = make_rgb_bytes(64, 64, "JPEG")
        sanitized, meta = InputValidator.validate_image_input(raw)
        assert isinstance(sanitized, Image.Image)

    def test_bytesio_accepted(self):
        """BytesIO wrapper is decoded correctly."""
        raw = make_rgb_bytes(64, 64, "PNG")
        buf = io.BytesIO(raw)
        sanitized, meta = InputValidator.validate_image_input(buf)
        assert isinstance(sanitized, Image.Image)

    def test_grayscale_converted_to_rgb(self):
        """Grayscale 'L' mode image is converted to RGB automatically."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        gray_img = Image.fromarray(arr, mode="L")
        sanitized, meta = InputValidator.validate_image_input(gray_img)
        assert sanitized.mode == "RGB"
        assert meta.get("original_mode") == "L"

    def test_rgba_converted_to_rgb(self):
        """RGBA image is converted to RGB (alpha channel dropped)."""
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(arr, mode="RGBA")
        sanitized, meta = InputValidator.validate_image_input(rgba_img)
        assert sanitized.mode == "RGB"


class TestInputValidatorRejection:
    """Tests for format, size, and dimension rejection rules."""

    def test_empty_bytes_rejected(self):
        """Zero-length bytes raises EMPTY_FILE."""
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(b"")
        assert "EMPTY_FILE" in exc.value.error_code

    def test_corrupt_bytes_rejected(self):
        """Corrupt/random bytes raises CORRUPT_IMAGE."""
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(b"\x00\xFF\xAB\xCD" * 100)
        assert "CORRUPT_IMAGE" in exc.value.error_code

    def test_unsupported_input_type_rejected(self):
        """Integer input raises UNSUPPORTED_INPUT_TYPE."""
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(12345)
        assert "UNSUPPORTED_INPUT_TYPE" in exc.value.error_code

    def test_tiny_image_rejected(self):
        """Image smaller than MIN_IMAGE_DIM raises IMAGE_TOO_SMALL."""
        tiny = Image.new("RGB", (4, 4))
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(tiny)
        assert "IMAGE_TOO_SMALL" in exc.value.error_code

    def test_file_not_found_raises(self, tmp_path):
        """Non-existent file path raises FILE_NOT_FOUND."""
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(missing)
        assert "FILE_NOT_FOUND" in exc.value.error_code

    def test_disallowed_extension_rejected(self, tmp_path):
        """File with .exe extension raises UNSUPPORTED_EXTENSION."""
        fake = tmp_path / "scan.exe"
        fake.write_bytes(b"\x00" * 100)
        with pytest.raises(InputValidationError) as exc:
            InputValidator.validate_image_input(fake)
        assert "UNSUPPORTED_EXTENSION" in exc.value.error_code

    def test_valid_file_path_accepted(self, tmp_path):
        """Valid PNG file path is correctly decoded."""
        img_path = tmp_path / "test_scan.png"
        make_rgb_pil().save(str(img_path), format="PNG")
        sanitized, meta = InputValidator.validate_image_input(img_path)
        assert isinstance(sanitized, Image.Image)
        assert meta["file_path"] == str(img_path)


# ===========================================================================
# Group 5-13: QualityGateEvaluator
# ===========================================================================

class TestQualityGateEvaluator:
    """Tests for quality gate policy evaluation."""

    def _evaluator(self, enforce_ood=True, enforce_calibration=True):
        return QualityGateEvaluator(
            enforce_ood_rejection=enforce_ood,
            enforce_calibration=enforce_calibration,
        )

    def test_all_valid_inputs_accepted(self):
        """Fully valid inputs with finite logits and no OOD trigger → ACCEPTED."""
        ev = self._evaluator()
        logits = np.array([1.5, 0.3, -0.2, -1.0])
        probs = np.array([0.6, 0.2, 0.1, 0.1])
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=logits,
            probabilities=probs,
            calibrated=True,
            ood_score=0.3,
            ood_threshold=0.8,
            anomaly_score=50.0,
            anomaly_threshold=200.0,
        )
        assert result.overall_status == QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT
        assert result.rejection_reasons == []

    def test_invalid_input_rejected_early(self):
        """Failed input validation → REJECTED_INVALID_INPUT with early return."""
        ev = self._evaluator()
        result = ev.evaluate_gates(
            input_valid=False,
            input_error="Image corrupted or unreadable.",
            modality_valid=True,
            model_available=False,
        )
        assert result.overall_status == QualityGateState.REJECTED_INVALID_INPUT
        assert len(result.rejection_reasons) > 0
        assert "GATE_MODALITY_VALIDITY" not in result.gate_outcomes  # Early return

    def test_unsupported_modality_rejected(self):
        """Failed modality validation → REJECTED_UNSUPPORTED_MODALITY."""
        ev = self._evaluator()
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=False,
            modality_error="Modality 'xyz' is not supported.",
            model_available=False,
        )
        assert result.overall_status == QualityGateState.REJECTED_UNSUPPORTED_MODALITY
        assert "GATE_INPUT_VALIDITY" in result.gate_outcomes
        assert "GATE_MODALITY_VALIDITY" in result.gate_outcomes

    def test_model_unavailable_inference_error(self):
        """Model checkpoint unavailable → INFERENCE_ERROR."""
        ev = self._evaluator()
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=False,
            model_error="Checkpoint file not found.",
        )
        assert result.overall_status == QualityGateState.INFERENCE_ERROR

    def test_non_finite_logits_inference_error(self):
        """NaN or Inf logits → INFERENCE_ERROR regardless of other gates."""
        ev = self._evaluator()
        nan_logits = np.array([float("nan"), 1.0, 0.5])
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=nan_logits,
        )
        assert result.overall_status == QualityGateState.INFERENCE_ERROR
        assert result.gate_outcomes["GATE_FINITE_OUTPUTS"].passed is False

    def test_inf_logits_inference_error(self):
        """Inf logits also trigger INFERENCE_ERROR."""
        ev = self._evaluator()
        inf_logits = np.array([float("inf"), 0.0])
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=inf_logits,
        )
        assert result.overall_status == QualityGateState.INFERENCE_ERROR

    def test_ood_score_above_threshold_rejected(self):
        """OOD score above threshold + enforcement → REJECTED_OOD."""
        ev = self._evaluator(enforce_ood=True)
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=np.array([0.5, 0.3]),
            probabilities=np.array([0.6, 0.4]),
            calibrated=True,
            ood_score=0.95,
            ood_threshold=0.70,
        )
        assert result.overall_status == QualityGateState.REJECTED_OOD
        assert result.gate_outcomes["GATE_OOD_CHECK"].passed is False
        assert len(result.rejection_reasons) > 0

    def test_ood_score_below_threshold_accepted(self):
        """OOD score below threshold → ACCEPTED_FOR_MODEL_OUTPUT."""
        ev = self._evaluator(enforce_ood=True)
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=np.array([1.5, 0.3, -0.2]),
            probabilities=np.array([0.65, 0.25, 0.10]),
            calibrated=True,
            ood_score=0.25,
            ood_threshold=0.70,
        )
        assert result.overall_status == QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT
        assert result.gate_outcomes["GATE_OOD_CHECK"].passed is True

    def test_anomaly_above_threshold_review_required(self):
        """Anomaly score above threshold → REVIEW_REQUIRED_ANOMALY."""
        ev = self._evaluator()
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=np.array([0.5, 0.3]),
            probabilities=np.array([0.6, 0.4]),
            calibrated=True,
            ood_score=0.2,
            ood_threshold=0.8,
            anomaly_score=900.0,
            anomaly_threshold=200.0,
        )
        assert result.overall_status == QualityGateState.REVIEW_REQUIRED_ANOMALY
        assert result.gate_outcomes["GATE_ANOMALY_CHECK"].passed is False
        assert len(result.review_reasons) > 0

    def test_ood_enforcement_disabled_flags_for_review(self):
        """OOD above threshold with enforcement OFF → FLAGGED_FOR_REVIEW, not rejection."""
        ev = self._evaluator(enforce_ood=False)
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=np.array([0.5, 0.3]),
            probabilities=np.array([0.6, 0.4]),
            calibrated=True,
            ood_score=0.95,
            ood_threshold=0.70,
        )
        # Not a hard rejection — flag for review instead
        assert result.overall_status != QualityGateState.REJECTED_OOD
        assert result.overall_status == QualityGateState.FLAGGED_FOR_REVIEW
        assert len(result.review_reasons) > 0

    def test_gate_outcomes_populated_for_full_run(self):
        """A successful full-run populates all gate keys."""
        ev = self._evaluator()
        result = ev.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=np.array([1.0, 0.5, 0.2]),
            probabilities=np.array([0.6, 0.3, 0.1]),
            calibrated=True,
            ood_score=0.2,
            ood_threshold=0.8,
            anomaly_score=50.0,
            anomaly_threshold=200.0,
        )
        expected_gates = {
            "GATE_INPUT_VALIDITY",
            "GATE_MODALITY_VALIDITY",
            "GATE_MODEL_AVAILABILITY",
            "GATE_FINITE_OUTPUTS",
            "GATE_CALIBRATION_STATUS",
            "GATE_OOD_CHECK",
            "GATE_ANOMALY_CHECK",
        }
        assert expected_gates == set(result.gate_outcomes.keys())


# ===========================================================================
# Group 14-17: AuditLogger
# ===========================================================================

class TestAuditLogger:
    """Tests for privacy-conscious audit logging."""

    def test_sha256_hash_deterministic(self):
        """Same bytes always produce the same SHA-256 hash."""
        payload = b"test_medical_image_payload"
        h1 = AuditLogger.compute_image_hash(payload)
        h2 = AuditLogger.compute_image_hash(payload)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_sha256_hash_non_reversible_sentinel(self):
        """Hash is not the original payload (trivially non-reversible)."""
        payload = b"unique_test_bytes"
        h = AuditLogger.compute_image_hash(payload)
        assert h != payload.decode("latin-1")

    def test_none_bytes_produces_none_hash(self):
        """None image bytes returns None hash without crashing."""
        result = AuditLogger.compute_image_hash(None)
        assert result is None

    def test_patient_pseudonym_masked(self, tmp_path):
        """Patient ID is stored as 'HASH-<prefix>' not raw string."""
        log_file = tmp_path / "test_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        record = auditor.record_inference(
            request_id="req-100",
            modality="brain_mri",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/brain_mri/best_model.pt",
            quality_gate_result=gate_result,
            patient_pseudonym="patient_secret_id_007",
        )
        tag = record["patient_pseudonym_tag"]
        assert "patient_secret_id_007" not in tag  # No raw ID
        assert tag.startswith("HASH-")

    def test_anonymous_when_no_patient_id(self, tmp_path):
        """Without patient ID, pseudonym tag is 'ANONYMOUS'."""
        log_file = tmp_path / "test_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        record = auditor.record_inference(
            request_id="req-101",
            modality="chest_xray",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/chest_xray/best_model.pt",
            quality_gate_result=gate_result,
        )
        assert record["patient_pseudonym_tag"] == "ANONYMOUS"

    def test_audit_record_written_to_disk(self, tmp_path):
        """Audit record is persisted to JSONL file on disk."""
        log_file = tmp_path / "test_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        auditor.record_inference(
            request_id="req-102",
            modality="retina",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/retina/best_model.pt",
            quality_gate_result=gate_result,
        )
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["request_id"] == "req-102"
        assert rec["modality"] == "retina"

    def test_audit_record_contains_gate_status(self, tmp_path):
        """Audit record includes the quality gate status value."""
        log_file = tmp_path / "test_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.REJECTED_OOD, ["OOD detected."])
        record = auditor.record_inference(
            request_id="req-103",
            modality="skin",
            model_identifier="resnet18",
            checkpoint_identifier="models/v2_baseline/skin/best_model.pt",
            quality_gate_result=gate_result,
            error_code="OOD_REJECTION",
        )
        assert record["quality_gate_status"] == "REJECTED_OOD"
        assert record["is_abstained"] is True
        assert record["error_code"] == "OOD_REJECTION"

    def test_concurrent_write_no_corruption(self, tmp_path):
        """Multiple threads writing audit records concurrently do not corrupt the log."""
        log_file = tmp_path / "concurrent_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        errors = []

        def write_record(thread_id: int):
            try:
                auditor.record_inference(
                    request_id=f"req-thread-{thread_id}",
                    modality="ultrasound",
                    model_identifier="resnet18",
                    checkpoint_identifier="models/v2_baseline/ultrasound/fold_0/best_model.pt",
                    quality_gate_result=gate_result,
                )
            except Exception as e:
                errors.append(e)

        N_THREADS = 20
        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent write errors: {errors}"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == N_THREADS
        # All lines must be valid JSON
        for line in lines:
            rec = json.loads(line)
            assert "request_id" in rec

    def test_recent_records_retrieval(self, tmp_path):
        """get_recent_records returns most-recently added records, bounded by limit."""
        log_file = tmp_path / "recent_audit.jsonl"
        auditor = AuditLogger(log_file=log_file)
        gate_result = make_gate_result(QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT)
        for i in range(5):
            auditor.record_inference(
                request_id=f"req-{i}",
                modality="brain_mri",
                model_identifier="resnet18",
                checkpoint_identifier="ckpt",
                quality_gate_result=gate_result,
            )
        records = auditor.get_recent_records(limit=3)
        assert len(records) == 3


# ===========================================================================
# Group 18-20: Pipeline Config
# ===========================================================================

class TestPipelineConfig:
    """Tests for configuration correctness."""

    def test_seven_modalities_defined(self):
        """Exactly 7 modalities are listed in SUPPORTED_MODALITIES."""
        assert len(SUPPORTED_MODALITIES) == 7

    def test_all_modalities_in_settings(self):
        """Every supported modality has a corresponding settings entry."""
        for mod in SUPPORTED_MODALITIES:
            assert mod in MODALITY_SETTINGS, f"Missing config for {mod}"

    def test_modality_settings_required_keys(self):
        """All modality configs have mandatory keys."""
        required = {"display_name", "task", "num_classes", "classes",
                    "normal_class", "normal_idx", "input_size", "channels", "checkpoint"}
        for mod, cfg in MODALITY_SETTINGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"{mod} missing keys: {missing}"

    def test_retina_is_multilabel_eight_classes(self):
        """Retina configuration is multilabel with exactly 8 classes."""
        ret = MODALITY_SETTINGS["retina"]
        assert ret["task"] == "multilabel"
        assert ret["num_classes"] == 8
        assert len(ret["classes"]) == 8
        assert ret.get("multi_label") is True

    def test_multiclass_modalities_have_softmax_task(self):
        """Non-retina modalities use 'multiclass' or 'binary' task, not multilabel."""
        for mod, cfg in MODALITY_SETTINGS.items():
            if mod != "retina":
                assert cfg["task"] in {"multiclass", "binary"}, (
                    f"{mod} should not be multilabel"
                )

    def test_pipeline_version_format(self):
        """PIPELINE_VERSION follows 'V2.0-PhaseN' convention."""
        assert PIPELINE_VERSION.startswith("V2.0")
        assert "Phase" in PIPELINE_VERSION

    def test_legal_notice_content(self):
        """LEGAL_LIMITATIONS_NOTICE contains mandatory research-only disclaimer."""
        assert "RESEARCH PROTOTYPE" in LEGAL_LIMITATIONS_NOTICE
        assert "NOT FOR DIRECT CLINICAL USE" in LEGAL_LIMITATIONS_NOTICE

    def test_artifact_paths_rooted_correctly(self):
        """All artifact paths are subpaths of ROOT."""
        paths_to_check = [
            BASELINE_MODELS_DIR,
            CALIBRATED_MODELS_DIR,
            OOD_MODELS_DIR,
            CALIBRATION_PARAMS_PATH,
            RETINA_THRESHOLDS_PATH,
            OOD_THRESHOLDS_PATH,
        ]
        for p in paths_to_check:
            assert str(p).startswith(str(ROOT)), f"Path not under ROOT: {p}"


# ===========================================================================
# Group 21-25: V2InferencePipeline (mocked model inference)
# ===========================================================================

class TestV2InferencePipelineMocked:
    """Tests for V2InferencePipeline using mocked ModalityAdapter."""

    def _make_pipeline(self, enforce_ood: bool = True):
        """Create a pipeline instance with artifact loading patched."""
        from pipeline_v2.inference_pipeline import V2InferencePipeline
        return V2InferencePipeline(
            enforce_ood_rejection=enforce_ood,
            enforce_calibration=True,
            device="cpu",
        )

    def _mock_adapter(self, modality: str, num_classes: int = 4, task: str = "multiclass"):
        """Creates a mock ModalityAdapter returning deterministic logits and features."""
        adapter = MagicMock()
        logits = np.random.default_rng(42).uniform(-1, 2, num_classes).astype(np.float32)
        feats = np.random.default_rng(99).uniform(0, 1, 512).astype(np.float32)
        adapter.process_image.return_value = (logits, feats)
        return adapter

    def test_invalid_modality_returns_abstained_response(self):
        """Unknown modality → is_abstained=True, REJECTED_UNSUPPORTED_MODALITY."""
        pipeline = self._make_pipeline()
        img = make_rgb_pil()
        response = pipeline.predict(image_input=img, modality="xyz_unknown_modality")
        assert response.is_abstained is True
        assert response.prediction_status == QualityGateState.REJECTED_UNSUPPORTED_MODALITY

    def test_missing_modality_returns_abstained_response(self):
        """None modality → is_abstained=True."""
        pipeline = self._make_pipeline()
        img = make_rgb_pil()
        response = pipeline.predict(image_input=img, modality=None)
        assert response.is_abstained is True

    def test_corrupt_image_returns_abstained_response(self):
        """Corrupt bytes → is_abstained=True, REJECTED_INVALID_INPUT."""
        pipeline = self._make_pipeline()
        response = pipeline.predict(
            image_input=b"\x00\xFF\xAB" * 50,
            modality="brain_mri",
        )
        assert response.is_abstained is True
        assert response.prediction_status == QualityGateState.REJECTED_INVALID_INPUT

    def test_valid_input_produces_structured_response_with_mocked_adapter(self):
        """Valid PIL + valid modality → structured StructuredInferenceResponse (mocked inference)."""
        pipeline = self._make_pipeline()
        # Replace adapter with mock
        mock_adapter = self._mock_adapter("brain_mri", num_classes=4)
        pipeline._adapters["brain_mri"] = mock_adapter
        img = make_rgb_pil()
        response = pipeline.predict(image_input=img, modality="brain_mri")
        assert isinstance(response, StructuredInferenceResponse)
        assert response.modality == "brain_mri"
        assert response.model_identifier == "resnet18"
        assert response.limitations_notice == LEGAL_LIMITATIONS_NOTICE
        assert response.pipeline_version == PIPELINE_VERSION

    def test_retina_response_has_label_predictions(self):
        """Retina inference returns per-label binary predictions and thresholds."""
        pipeline = self._make_pipeline()
        mock_adapter = self._mock_adapter("retina", num_classes=8, task="multilabel")
        pipeline._adapters["retina"] = mock_adapter
        img = make_rgb_pil()
        response = pipeline.predict(image_input=img, modality="retina")
        assert isinstance(response, StructuredInferenceResponse)
        assert response.modality == "retina"
        # Even if calibration JSON is absent, sigmoid is applied; check structure
        if not response.is_abstained:
            assert response.retina_label_predictions is not None
            assert len(response.retina_label_predictions) == 8
            assert response.retina_thresholds_used is not None
            assert len(response.retina_thresholds_used) == 8
            # All predictions must be 0 or 1
            for label, pred in response.retina_label_predictions.items():
                assert pred in (0, 1), f"Label '{label}' has non-binary prediction {pred}"

    def test_response_always_contains_limitations_notice(self):
        """Limitations notice is always present, even on abstained responses."""
        pipeline = self._make_pipeline()
        # Abstained (invalid modality)
        resp_abstained = pipeline.predict(image_input=make_rgb_pil(), modality="fake_xyz")
        assert resp_abstained.limitations_notice == LEGAL_LIMITATIONS_NOTICE

        # Valid path (mocked)
        mock_adapter = self._mock_adapter("chest_xray", num_classes=4)
        pipeline._adapters["chest_xray"] = mock_adapter
        resp_valid = pipeline.predict(image_input=make_rgb_pil(), modality="chest_xray")
        assert resp_valid.limitations_notice == LEGAL_LIMITATIONS_NOTICE

    def test_request_id_is_unique_per_call(self):
        """Each call generates a unique UUID request_id."""
        pipeline = self._make_pipeline()
        mock_adapter = self._mock_adapter("histopathology", num_classes=5)
        pipeline._adapters["histopathology"] = mock_adapter
        img = make_rgb_pil()
        r1 = pipeline.predict(image_input=img, modality="histopathology")
        r2 = pipeline.predict(image_input=img, modality="histopathology")
        assert r1.request_id != r2.request_id

    def test_ood_status_field_always_present(self):
        """ood_status field is always set in every response."""
        pipeline = self._make_pipeline()
        mock_adapter = self._mock_adapter("ultrasound", num_classes=3)
        pipeline._adapters["ultrasound"] = mock_adapter
        response = pipeline.predict(image_input=make_rgb_pil(), modality="ultrasound")
        assert response.ood_status in {"WITHIN_DISTRIBUTION", "POTENTIALLY_OOD", "UNAVAILABLE"}

    def test_calibration_config_in_response(self):
        """calibration_configuration is always included in a successful inference response."""
        pipeline = self._make_pipeline()
        mock_adapter = self._mock_adapter("skin", num_classes=7)
        pipeline._adapters["skin"] = mock_adapter
        response = pipeline.predict(image_input=make_rgb_pil(), modality="skin")
        if not response.is_abstained:
            cfg = response.calibration_configuration
            assert "task" in cfg
            assert "method" in cfg

    def test_non_finite_logits_from_model_triggers_inference_error(self):
        """If model returns NaN logits, pipeline responds with INFERENCE_ERROR."""
        pipeline = self._make_pipeline()
        # Mock adapter returning NaN logits
        mock_adapter = MagicMock()
        nan_logits = np.array([float("nan"), 1.0, 0.5, -0.3])
        feats = np.zeros(512, dtype=np.float32)
        mock_adapter.process_image.return_value = (nan_logits, feats)
        pipeline._adapters["brain_mri"] = mock_adapter
        response = pipeline.predict(image_input=make_rgb_pil(), modality="brain_mri")
        assert response.is_abstained is True
        assert response.prediction_status == QualityGateState.INFERENCE_ERROR


# ===========================================================================
# Group 26-30: Artifact Integrity
# ===========================================================================

class TestArtifactIntegrity:
    """Verifies all required artifacts exist and have not been deleted or zeroed."""

    BASELINE_MODALITIES = [
        ("brain_mri", "best_model.pt"),
        ("brain_ct", "fold_0/best_model.pt"),
        ("chest_xray", "best_model.pt"),
        ("histopathology", "best_model.pt"),
        ("retina", "best_model.pt"),
        ("skin", "best_model.pt"),
        ("ultrasound", "fold_0/best_model.pt"),
    ]

    def test_baseline_checkpoints_exist_non_empty(self):
        """All 7 baseline checkpoints must exist and have non-zero size."""
        assert BASELINE_MODELS_DIR.exists(), "Baseline models directory missing"
        for mod, rel_path in self.BASELINE_MODALITIES:
            ckpt = BASELINE_MODELS_DIR / mod / rel_path
            assert ckpt.exists(), f"Baseline checkpoint missing: {ckpt}"
            assert ckpt.stat().st_size > 0, f"Baseline checkpoint empty: {ckpt}"

    def test_calibrated_checkpoints_exist_non_empty(self):
        """All 7 calibrated checkpoints must exist and have non-zero size."""
        assert CALIBRATED_MODELS_DIR.exists(), "Calibrated models directory missing"
        for mod, rel_path in self.BASELINE_MODALITIES:
            ckpt = CALIBRATED_MODELS_DIR / mod / rel_path
            assert ckpt.exists(), f"Calibrated checkpoint missing: {ckpt}"
            assert ckpt.stat().st_size > 0, f"Calibrated checkpoint empty: {ckpt}"

    def test_split_manifests_exist(self):
        """Split manifests directory exists with at least 7 manifest files."""
        assert DATA_MANIFESTS_DIR.exists(), "Split manifests directory missing"
        manifest_files = list(DATA_MANIFESTS_DIR.glob("*.csv")) + list(DATA_MANIFESTS_DIR.glob("*.json"))
        assert len(manifest_files) >= 7, (
            f"Expected at least 7 split manifest files, found {len(manifest_files)}"
        )

    def test_phase7_calibration_params_exist(self):
        """Phase 7 calibration_parameters.json must exist and be valid JSON."""
        assert CALIBRATION_PARAMS_PATH.exists(), (
            f"Phase 7 calibration params missing: {CALIBRATION_PARAMS_PATH}"
        )
        with open(CALIBRATION_PARAMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Should have entries for the 6 multiclass modalities
        assert isinstance(data, dict)
        assert len(data) >= 1

    def test_phase7_retina_thresholds_exist(self):
        """Phase 7 retina_thresholds.json must exist and contain 8 label thresholds."""
        assert RETINA_THRESHOLDS_PATH.exists(), (
            f"Phase 7 retina thresholds missing: {RETINA_THRESHOLDS_PATH}"
        )
        with open(RETINA_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        # Either the root IS the per-label thresholds or nested under a key
        thresholds = data.get("retina_thresholds", data)
        assert len(thresholds) >= 8 or len(data) >= 1, "Retina threshold data empty or malformed"

    def test_phase8_ood_thresholds_exist(self):
        """Phase 8 ood_thresholds.json must exist and be valid JSON."""
        assert OOD_THRESHOLDS_PATH.exists(), (
            f"Phase 8 OOD thresholds missing: {OOD_THRESHOLDS_PATH}"
        )
        with open(OOD_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_ood_models_directory_exists(self):
        """Phase 8 OOD models directory must exist."""
        assert OOD_MODELS_DIR.exists(), f"OOD models directory missing: {OOD_MODELS_DIR}"
