"""
api/schemas.py — Module 21: Production REST API
Pydantic V2 schemas for request and response validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Error Response ─────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = Field(default=None, description="Location of error in request")
    msg: str = Field(description="Error message")
    type: Optional[str] = Field(default=None, description="Error type code")


class ErrorResponse(BaseModel):
    error_code: str = Field(description="Categorical error code")
    message: str = Field(description="Human-readable error summary")
    request_id: Optional[str] = Field(default="N/A", description="Request correlation tracing ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp"
    )
    path: Optional[str] = Field(default=None, description="Request path")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed error breakdown")


# ── Prediction Schemas ────────────────────────────────────────────────────────

class TopPredictionItem(BaseModel):
    disease: str = Field(description="Disease or condition name")
    confidence: float = Field(description="Class probability [0.0 - 1.0]")


class GradCAMMetadata(BaseModel):
    generated: bool = Field(description="Whether GradCAM heatmap was generated")
    heatmap_path: Optional[str] = Field(default=None, description="Path to generated heatmap image")
    target_layer: Optional[str] = Field(default=None, description="Model target layer used for GradCAM")


class AnomalyResultSchema(BaseModel):
    anomaly_score: float = Field(description="Calculated Mahalanobis / OOD anomaly score")
    anomaly_threshold: float = Field(description="Decision threshold for anomaly classification")
    is_anomaly: bool = Field(description="Whether scan is classified as out-of-distribution anomaly")
    anomaly_status: str = Field(description="Pattern classification: KNOWN_DISEASE or UNKNOWN_ANOMALY")


class ExecutionTimingsSchema(BaseModel):
    input_validation_ms: Optional[float] = None
    preprocessing_ms: Optional[float] = None
    modality_classification_ms: Optional[float] = None
    routing_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    confidence_ms: Optional[float] = None
    gradcam_ms: Optional[float] = None
    anomaly_ms: Optional[float] = None
    total_execution_ms: float = Field(description="Total pipeline wall-clock time in milliseconds")


class PredictionResponse(BaseModel):
    request_id: str = Field(description="Unique session / prediction UUID")
    patient_id: str = Field(description="Patient / Case ID")
    modality: str = Field(description="Detected or target imaging modality")
    modality_confidence: float = Field(default=1.0, description="Modality router confidence score")
    routing_mode: str = Field(default="automatic", description="Routing mode: automatic or manual_override")
    predicted_class: str = Field(description="Primary predicted disease / condition")
    confidence_score: float = Field(description="Diagnostic confidence score [0.0 - 1.0]")
    reliability: str = Field(description="Reliability rating: HIGH, MODERATE, or LOW")
    requires_review: bool = Field(description="Whether manual physician review is recommended")
    risk_level: str = Field(default="LOW", description="Clinical risk rating: LOW, MODERATE, HIGH, CRITICAL")
    primary_diagnosis: str = Field(description="Formal primary diagnosis statement")
    clinical_note: Optional[str] = Field(default=None, description="Automated clinical note")
    differential_diagnoses: List[Any] = Field(default_factory=list, description="Top alternative predictions")
    top_predictions: List[TopPredictionItem] = Field(default_factory=list, description="Ranked class predictions")
    gradcam: GradCAMMetadata = Field(description="GradCAM visual explainability metadata")
    anomaly: AnomalyResultSchema = Field(description="Out-of-distribution anomaly detection metrics")
    timings: ExecutionTimingsSchema = Field(description="Pipeline latency timings")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of prediction execution"
    )


class BatchPredictionResponse(BaseModel):
    total_images: int = Field(description="Total uploaded images processed")
    successful: int = Field(description="Number of successful predictions")
    failed: int = Field(description="Number of failed predictions")
    total_ms: float = Field(description="Total batch processing latency in milliseconds")
    results: List[PredictionResponse] = Field(description="List of individual prediction responses")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of failed item errors")


# ── History Schemas ───────────────────────────────────────────────────────────

class HistoryRecordResponse(BaseModel):
    prediction_id: str
    created_at: str
    patient_id: str
    case_id: str
    modality: str
    predicted_disease: str
    confidence: float
    confidence_pct: float
    explainability_summary: Optional[str] = ""
    heatmap_path: Optional[str] = ""
    anomaly_status: str
    risk_level: str
    risk_score: float
    recommended_action: str
    report_path: Optional[str] = ""
    processing_time_ms: float
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoryListResponse(BaseModel):
    total_records: int = Field(description="Total matching prediction records in database")
    limit: int = Field(description="Applied page limit")
    offset: int = Field(description="Applied page offset")
    records: List[HistoryRecordResponse] = Field(description="List of prediction history records")


# ── Report Schemas ────────────────────────────────────────────────────────────

class ReportMetadataResponse(BaseModel):
    prediction_id: str = Field(description="Target prediction ID")
    patient_id: str = Field(description="Patient ID")
    generated_at: str = Field(description="Report generation timestamp")
    modality: str = Field(description="Imaging modality")
    primary_diagnosis: str = Field(description="Primary diagnosis")
    confidence: str = Field(description="Confidence percentage label")
    reliability: str = Field(description="Diagnostic reliability level")
    clinical_note: str = Field(description="Narrative clinical note")
    differential_diagnoses: List[Any] = Field(default_factory=list)
    requires_manual_review: bool
    anomaly_score: float
    is_anomaly: bool
    processing_time_ms: float
    download_urls: Dict[str, str] = Field(
        description="Direct download URLs for json, md, and pdf formats"
    )


# ── Benchmark Schemas ─────────────────────────────────────────────────────────

class ModalityBenchmarkModel(BaseModel):
    modality: str
    accuracy: float
    f1: float
    auc: float
    inference_ms: int


class BenchmarkSummaryResponse(BaseModel):
    platform_version: str
    benchmark_date: str
    overall_accuracy: float
    overall_auc: float
    total_test_samples: int
    anomaly_detection_auc: float
    models: List[ModalityBenchmarkModel]


# ── System / Health Schemas ───────────────────────────────────────────────────

class ReadinessStatusResponse(BaseModel):
    status: str = Field(description="Readiness status: READY or NOT_READY")
    engine_ready: bool = Field(description="Whether MedicalAIEngine is initialized and ready")
    database_ready: bool = Field(description="Whether SQLite database scope is responsive")


class HealthStatusResponse(BaseModel):
    status: str = Field(description="Platform status: HEALTHY, DEGRADED, or ERROR")
    api_version: str = Field(description="REST API version string")
    engine_state: str = Field(description="MedicalAIEngine state string")
    database_status: str = Field(description="Database connectivity status: OK or ERROR")
    loaded_adapters: List[str] = Field(description="List of registered modality adapters")
    uptime_seconds: float = Field(description="Process uptime in seconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SystemInfoResponse(BaseModel):
    platform_name: str
    platform_version: str
    supported_modalities: List[str]
    pipeline_stages: List[str]
    database_backend: str
    xai_framework: str
    anomaly_engine: str
    tech_stack: List[str]
