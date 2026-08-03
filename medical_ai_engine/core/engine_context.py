"""
Medical AI Engine — EngineContext

Single shared state carrier passed through every pipeline stage.
Exactly as designed in Module 14.1.5 Architecture Specification.

No pipeline stage should receive dozens of independent parameters.
Every stage receives and mutates exactly one EngineContext instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from medical_ai_engine.core.engine_state import EngineState


@dataclass
class EngineContext:
    """
    Shared state carrier for the complete orchestration pipeline.

    Passed by reference through every BasePipelineStage.process(ctx) call.
    Accumulates state progressively: each stage populates its relevant fields.

    Fields are organized by lifecycle phase:
        Identity       → request_id, session_id
        Input          → image, image_source, manual_modality_override
        Metadata       → metadata
        Timing         → timestamps
        State          → execution_state
        Routing        → modality_prediction
        Prediction     → disease_prediction, prediction_result
        Confidence     → confidence
        Reporting      → clinical_report
        Extensions     → gradcam_result, anomaly_result, future_extensions
        Audit          → logs, warnings, errors, performance_metrics
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str
    """Unique identifier for this prediction request (UUID4 format)."""

    session_id: str
    """Unique identifier for the execution session wrapping this request."""

    # ── Input ─────────────────────────────────────────────────────────────────
    image: Optional[Any] = None
    """
    PIL.Image.Image loaded and validated at the ingestion stage.
    Type is Optional[Any] to avoid importing PIL at this level.
    """

    image_source: Optional[Union[str, Path]] = None
    """Original image file path or URI provided in the request."""

    image_bytes: Optional[bytes] = None
    """Raw image bytes (set if input was provided as bytes)."""

    manual_modality_override: Optional[str] = None
    """
    If set, skips Module 6 modality detection and routes directly.
    Valid values: "brain_mri", "chest_xray", "brain_ct",
                  "histopathology", "retina", "skin", "ultrasound"
    """

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    General key-value metadata populated across pipeline stages.
    Expected keys (populated progressively):
        "format"           → image file format (JPEG, PNG, etc.)
        "size"             → (width, height) tuple
        "color_mode"       → "RGB", "L", etc.
        "quality_score"    → image quality score (0.0–1.0)
        "selected_adapter" → chosen modality adapter class name
        "inference_backend"→ "pytorch", "onnx", "torchscript"
    """

    # ── Timing ────────────────────────────────────────────────────────────────
    timestamps: Dict[str, str] = field(default_factory=dict)
    """
    ISO 8601 UTC timestamps recorded at each lifecycle stage.
    Expected keys:
        "request_received"       → EngineContext creation time
        "validation_complete"    → after ValidateInputStage
        "quality_check_complete" → after ImageQualityCheckStage
        "modality_predicted"     → after DetectModalityStage
        "adapter_selected"       → after RegistryLookupStage
        "disease_predicted"      → after ExecuteInferenceStage
        "confidence_analysed"    → after ConfidenceAnalysisStage
        "gradcam_complete"       → after GradCAMHookStage
        "anomaly_complete"       → after AnomalyHookStage
        "report_complete"        → after ClinicalReportHookStage
        "prediction_json_ready"  → after AssemblePredictionJSONStage
        "logged"                 → after LoggingStage
        "response_sent"          → after ResponseStage
    """

    # ── Execution State ───────────────────────────────────────────────────────
    execution_state: str = EngineState.CREATED.value
    """
    Current EngineState value as a string.
    Updated by the engine at each lifecycle transition.
    """

    # ── Modality Routing (populated by DetectModalityStage) ───────────────────
    modality_prediction: Optional[Dict[str, Any]] = None
    """
    Output of the modality routing stage.
    Schema:
        {
            "detected_modality": str,         # e.g. "skin"
            "modality_confidence": float,     # 0.0 – 1.0
            "routing_mode": str,              # "automatic" | "manual_override"
            "routing_time_ms": float,
            "low_confidence_warning": bool,   # true if confidence < 0.70
            "top_k_modalities": list          # top-k modality predictions
        }
    """

    # ── Disease Prediction (populated by ExecuteInferenceStage) ───────────────
    disease_prediction: Optional[Dict[str, Any]] = None
    """
    Raw output dict from the selected module adapter's predict() call.
    Schema varies slightly by module (Module 11 adds severity_grade,
    Modules 12/13 add prediction_reliability, requires_manual_review).
    """

    # ── Normalized Prediction Result ──────────────────────────────────────────
    prediction_result: Optional[Dict[str, Any]] = None
    """
    Normalized, schema-validated prediction result assembled by
    AssemblePredictionJSONStage. Conforms to UnifiedPredictionResponse spec.
    """

    # ── Confidence Analysis (populated by ConfidenceAnalysisStage) ───────────
    confidence: Optional[Dict[str, Any]] = None
    """
    Normalized confidence assessment.
    Schema:
        {
            "score": float,                 # top-1 confidence (0.0–1.0)
            "reliability": str,             # "high" | "moderate" | "low"
            "requires_manual_review": bool,
            "severity_grade": int | None,   # for Module 11 (retina)
        }
    """

    # ── Clinical Report (populated by ClinicalReportHookStage) ───────────────
    clinical_report: Optional[Dict[str, Any]] = None
    """
    Disease-specific clinical diagnostic report.
    Schema:
        {
            "report_type": str,
            "primary_diagnosis": str,
            "reliability": str,
            "requires_manual_review": bool,
            "differential_diagnoses": list[str],
            "clinical_note": str,
        }
    """

    # ── Grad-CAM Extension (Module 15) ────────────────────────────────────────
    gradcam_result: Optional[Dict[str, Any]] = None
    """
    Populated by GradCAMHookStage when enable_gradcam is True.
    Schema:
        {
            "heatmap_path": str,
            "target_layer": str,
            "cam_method": str,
            "generation_time_ms": float,
        }
    """

    # ── Anomaly Detection Extension (Module 16) ───────────────────────────────
    anomaly_result: Optional[Dict[str, Any]] = None
    """
    Populated by AnomalyHookStage when enable_anomaly_detection is True.
    Schema:
        {
            "anomaly_score": float,
            "is_anomaly": bool,
            "anomaly_threshold": float,
            "detection_time_ms": float,
        }
    """

    # ── Performance Metrics ───────────────────────────────────────────────────
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    """
    Per-stage timing metrics in milliseconds.
    Expected keys:
        "validation_ms", "quality_check_ms", "routing_ms",
        "adapter_lookup_ms", "inference_ms", "confidence_ms",
        "gradcam_ms", "anomaly_ms", "report_ms", "assembly_ms",
        "logging_ms", "total_ms"
    """

    # ── Audit Trail ───────────────────────────────────────────────────────────
    logs: List[str] = field(default_factory=list)
    """Append-only structured log entries. Each stage appends its messages."""

    warnings: List[str] = field(default_factory=list)
    """Non-fatal alerts (e.g. low routing confidence, image quality warning)."""

    errors: List[Dict[str, str]] = field(default_factory=list)
    """
    Non-fatal or recovered errors. Fatal errors raise exceptions instead.
    Schema per entry:
        {"stage": str, "error": str, "timestamp": str}
    """

    # ── Future Extensions ─────────────────────────────────────────────────────
    future_extensions: Dict[str, Any] = field(default_factory=dict)
    """
    Open namespace for Modules 15–19+ to store custom data
    without requiring EngineContext schema changes.

    Convention: use module-prefixed keys to avoid collisions.
    Examples:
        future_extensions["module_15_gradcam"] = {...}
        future_extensions["module_16_anomaly"] = {...}
        future_extensions["module_19_db_id"] = "uuid-..."
    """

    # ── Computed Properties ───────────────────────────────────────────────────

    def log(self, message: str, stage: Optional[str] = None) -> None:
        """Append a log entry with optional stage prefix."""
        prefix = f"[{stage}] " if stage else ""
        entry = f"{_utc_now()} {prefix}{message}"
        self.logs.append(entry)

    def warn(self, message: str, stage: Optional[str] = None) -> None:
        """Append a non-fatal warning."""
        prefix = f"[{stage}] " if stage else ""
        self.warnings.append(f"{_utc_now()} {prefix}WARNING: {message}")

    def add_error(self, stage: str, error: str) -> None:
        """Record a non-fatal error for a specific stage."""
        self.errors.append(
            {"stage": stage, "error": error, "timestamp": _utc_now()}
        )

    def set_timestamp(self, key: str) -> None:
        """Record the current UTC time for a named lifecycle event."""
        self.timestamps[key] = _utc_now()

    def record_metric(self, key: str, value_ms: float) -> None:
        """Record a performance metric in milliseconds."""
        self.performance_metrics[key] = round(value_ms, 3)

    def has_errors(self) -> bool:
        """Returns True if any non-fatal errors were recorded."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Returns True if any warnings were recorded."""
        return len(self.warnings) > 0

    def total_execution_ms(self) -> Optional[float]:
        """Returns total_ms from performance_metrics if available."""
        return self.performance_metrics.get("total_ms")

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dict for logging."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "execution_state": self.execution_state,
            "modality": (self.modality_prediction or {}).get("detected_modality"),
            "predicted_class": (self.disease_prediction or {}).get("predicted_class"),
            "confidence": (self.confidence or {}).get("score"),
            "total_ms": self.total_execution_ms(),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "log_entries": len(self.logs),
        }

    def __repr__(self) -> str:
        return (
            f"EngineContext("
            f"request_id={self.request_id!r}, "
            f"session_id={self.session_id!r}, "
            f"state={self.execution_state!r})"
        )


def _utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
