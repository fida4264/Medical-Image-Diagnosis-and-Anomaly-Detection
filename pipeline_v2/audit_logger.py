"""
pipeline_v2/audit_logger.py
===========================
Privacy-conscious audit logging and provenance tracking for Phase 9 inference.
Records cryptographically hashed image provenance, model checkpoints, gate decisions,
and execution parameters without exposing raw pixels or PHI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_v2.config import AUDIT_LOG_FILE, PIPELINE_VERSION
from pipeline_v2.schemas import QualityGateResult, QualityGateState

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Thread-safe, privacy-preserving audit logger for medical inference provenance.
    """

    def __init__(self, log_file: Path = AUDIT_LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._in_memory_records: List[Dict[str, Any]] = []

    @staticmethod
    def compute_image_hash(image_bytes: Optional[bytes]) -> Optional[str]:
        """Calculates SHA-256 hash of raw image payload for non-reversible provenance."""
        if not image_bytes:
            return None
        return hashlib.sha256(image_bytes).hexdigest()

    def record_inference(
        self,
        request_id: str,
        modality: str,
        model_identifier: str,
        checkpoint_identifier: str,
        quality_gate_result: QualityGateResult,
        image_bytes: Optional[bytes] = None,
        patient_pseudonym: Optional[str] = None,
        ood_method: Optional[str] = None,
        ood_score: Optional[float] = None,
        ood_threshold: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        anomaly_threshold: Optional[float] = None,
        preprocessing_config_id: str = "resnet18_standard_224_imagenet",
        calibration_artifact_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates, logs, and persists a structured audit record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        image_sha256 = self.compute_image_hash(image_bytes)

        # De-identify patient if provided
        patient_tag = "ANONYMOUS"
        if patient_pseudonym:
            # Store only a masked pseudonym
            patient_tag = f"HASH-{hashlib.sha256(patient_pseudonym.encode()).hexdigest()[:8]}"

        record: Dict[str, Any] = {
            "request_id": request_id,
            "timestamp": timestamp,
            "pipeline_version": PIPELINE_VERSION,
            "patient_pseudonym_tag": patient_tag,
            "image_sha256": image_sha256,
            "modality": modality,
            "model_identifier": model_identifier,
            "checkpoint_identifier": checkpoint_identifier,
            "preprocessing_config_id": preprocessing_config_id,
            "calibration_artifact_id": calibration_artifact_id or "none",
            "quality_gate_status": quality_gate_result.overall_status.value,
            "is_abstained": quality_gate_result.overall_status in (
                QualityGateState.REJECTED_OOD,
                QualityGateState.REJECTED_INVALID_INPUT,
                QualityGateState.REJECTED_UNSUPPORTED_MODALITY,
                QualityGateState.INFERENCE_ERROR,
            ),
            "rejection_reasons": quality_gate_result.rejection_reasons,
            "review_reasons": quality_gate_result.review_reasons,
            "gate_outcomes": {
                k: {"passed": v.passed, "message": v.message}
                for k, v in quality_gate_result.gate_outcomes.items()
            },
            "ood_metrics": {
                "method": ood_method,
                "score": round(ood_score, 4) if ood_score is not None else None,
                "threshold": round(ood_threshold, 4) if ood_threshold is not None else None,
            },
            "anomaly_metrics": {
                "score": round(anomaly_score, 4) if anomaly_score is not None else None,
                "threshold": round(anomaly_threshold, 4) if anomaly_threshold is not None else None,
            },
            "error_code": error_code,
        }

        # Thread-safe write to disk and memory
        try:
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                self._in_memory_records.append(record)
                if len(self._in_memory_records) > 200:
                    self._in_memory_records.pop(0)
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Raise to prevent silent un-audited inference
            raise RuntimeError(f"Audit log failure: {e}") from e

        return record

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent in-memory audit records."""
        with self._lock:
            return list(reversed(self._in_memory_records[-limit:]))
