"""
api/routes/prediction.py — Module 21: Production REST API
Prediction endpoints with strict file upload security and input validation.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from PIL import Image, ImageOps

from api.dependencies import get_engine, verify_api_key
from api.exceptions import (
    FileTooLargeException,
    InvalidImageException,
    PredictionFailedException,
    UnsupportedModalityException,
)
from api.schemas import (
    AnomalyResultSchema,
    BatchPredictionResponse,
    ExecutionTimingsSchema,
    GradCAMMetadata,
    PredictionResponse,
    TopPredictionItem,
)
from api.settings import settings
from medical_ai_engine import MedicalAIEngine
from medical_ai_engine.core.engine_context import EngineContext

router = APIRouter(prefix="", tags=["Prediction Engine"])

SUPPORTED_MODALITIES = {
    "brain_mri", "chest_xray", "brain_ct",
    "histopathology", "retina", "skin", "ultrasound"
}


def _validate_and_sanitize_image_file(file: UploadFile) -> str:
    """
    Strict multi-stage security validation for uploaded medical scans:
      1. Filename & extension check
      2. MIME type whitelist verification
      3. File size limit enforcement (max 20MB)
      4. PIL image header decoding & corruption check
    """
    if not file.filename:
        raise InvalidImageException("Uploaded file name is missing.")

    # 1. Extension Check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidImageException(
            f"File extension '{ext}' is unallowed. Allowed extensions: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )

    # 2. Content Read & Size Check
    contents = file.file.read()
    file_size = len(contents)
    if file_size == 0:
        raise InvalidImageException("Uploaded medical scan file is 0 bytes (empty).")

    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise FileTooLargeException(max_mb=max_mb)

    # 3. MIME Type Check
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in settings.ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        raise InvalidImageException(
            f"MIME type '{content_type}' is unallowed. Allowed image formats: PNG, JPEG, BMP, TIFF, DICOM."
        )

    # 4. Save & PIL Decoding Check
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        with Image.open(tmp_path) as img:
            img.verify()  # Header verification
        with Image.open(tmp_path) as img2:
            img2.load()    # Pixel decoding check
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise InvalidImageException(
            f"File header or pixel data is corrupted and cannot be decoded: {exc}"
        ) from exc

    return tmp_path


def _build_prediction_response(ctx: EngineContext, patient_id: Optional[str] = None) -> PredictionResponse:
    """Format EngineContext object into a standard PredictionResponse schema."""
    mod_info  = ctx.modality_prediction or {}
    dis_info  = ctx.disease_prediction or {}
    conf_info = ctx.confidence or {}
    cam_info  = ctx.gradcam_result or {}
    anom_info = ctx.anomaly_result or {}
    rpt_info  = ctx.clinical_report or {}

    modality      = str(mod_info.get("detected_modality", "unknown"))
    conf_score    = float(conf_info.get("score", dis_info.get("confidence", 0.0)))
    reliability   = str(conf_info.get("reliability", "low")).upper()
    req_review    = bool(conf_info.get("requires_manual_review", True))
    predicted_cls = str(dis_info.get("predicted_class", "unknown"))

    # Risk level calculation
    if reliability == "HIGH" and not req_review:
        risk_level = "LOW"
    elif reliability == "MODERATE":
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"
    if bool(anom_info.get("is_anomaly", False)):
        risk_level = "CRITICAL"

    # Top-K predictions
    top_items: List[TopPredictionItem] = []
    for item in dis_info.get("top_k_predictions", []):
        if isinstance(item, dict):
            top_items.append(TopPredictionItem(
                disease=str(item.get("disease", "unknown")),
                confidence=float(item.get("confidence", 0.0)),
            ))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            top_items.append(TopPredictionItem(
                disease=str(item[0]),
                confidence=float(item[1]),
            ))

    is_anomaly     = bool(anom_info.get("is_anomaly", False))
    anomaly_status = "UNKNOWN_ANOMALY" if is_anomaly else "KNOWN_DISEASE"

    total_ms = 0.0
    try:
        total_ms = float(ctx.total_execution_ms() or 0.0)
    except Exception:
        pass

    return PredictionResponse(
        request_id=ctx.request_id,
        patient_id=str(patient_id or (ctx.metadata or {}).get("patient_id", "PT-ANON-001")),
        modality=modality,
        modality_confidence=float(mod_info.get("modality_confidence", 1.0)),
        routing_mode=str(mod_info.get("routing_mode", "automatic")),
        predicted_class=predicted_cls,
        confidence_score=round(conf_score, 4),
        reliability=reliability,
        requires_review=req_review,
        risk_level=risk_level,
        primary_diagnosis=str(rpt_info.get("primary_diagnosis", predicted_cls)),
        clinical_note=rpt_info.get("clinical_note"),
        differential_diagnoses=list(rpt_info.get("differential_diagnoses", [])),
        top_predictions=top_items,
        gradcam=GradCAMMetadata(
            generated=bool(cam_info.get("heatmap_path")),
            heatmap_path=cam_info.get("heatmap_path"),
            target_layer=cam_info.get("target_layer"),
        ),
        anomaly=AnomalyResultSchema(
            anomaly_score=round(float(anom_info.get("anomaly_score", 0.0)), 4),
            anomaly_threshold=round(float(anom_info.get("anomaly_threshold", 0.5)), 2),
            is_anomaly=is_anomaly,
            anomaly_status=anomaly_status,
        ),
        timings=ExecutionTimingsSchema(
            total_execution_ms=round(total_ms, 2),
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _store_record_in_db(resp: PredictionResponse) -> None:
    """Best-effort database storage in Module 19 repository."""
    try:
        from medical_ai_engine.database.database_manager import DatabaseManager
        from medical_ai_engine.database.prediction_repository import PredictionRecord, PredictionRepository
        db = DatabaseManager()
        repo = PredictionRepository(db)
        record = PredictionRecord(
            prediction_id=resp.request_id,
            patient_id=resp.patient_id,
            case_id=f"CASE-{resp.request_id[:8]}",
            modality=resp.modality,
            predicted_disease=resp.predicted_class,
            confidence=resp.confidence_score,
            explainability_summary=f"GradCAM layer {resp.gradcam.target_layer or 'default'}",
            heatmap_path=resp.gradcam.heatmap_path or "",
            anomaly_status=resp.anomaly.anomaly_status,
            risk_level=resp.risk_level,
            risk_score=resp.anomaly.anomaly_score,
            recommended_action="CLINICAL_REVIEW" if resp.requires_review else "ROUTINE_FOLLOW_UP",
            report_path="",
            processing_time_ms=resp.timings.total_execution_ms,
        )
        repo.save(record)
    except Exception:
        pass


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Diagnostic Analysis on a Medical Image",
    description=(
        "Upload a medical scan image to run automated modality classification, "
        "disease diagnosis, GradCAM visual explainability, and out-of-distribution "
        "anomaly detection."
    ),
)
async def predict_single(
    file: UploadFile = File(..., description="Medical image file (PNG, JPG, BMP, TIFF)"),
    modality: Optional[str] = Form(None, description="Optional target modality override"),
    patient_id: Optional[str] = Form(None, description="Optional patient / case identifier"),
    engine: MedicalAIEngine = Depends(get_engine),
    _auth: Optional[str] = Depends(verify_api_key),
) -> PredictionResponse:
    if modality and modality.lower() not in SUPPORTED_MODALITIES and modality != "Auto Detect":
        raise UnsupportedModalityException(modality)

    tmp_path = _validate_and_sanitize_image_file(file)

    try:
        override = None if (not modality or modality == "Auto Detect") else modality.lower()
        extra_meta: Dict[str, Any] = {"patient_id": patient_id} if patient_id else {}

        ctx: EngineContext = engine.execute(
            image_source=tmp_path,
            manual_modality_override=override,
            extra_metadata=extra_meta,
        )
        resp = _build_prediction_response(ctx, patient_id=patient_id)
        _store_record_in_db(resp)
        return resp
    except (InvalidImageException, FileTooLargeException, UnsupportedModalityException, PredictionFailedException):
        raise
    except Exception as exc:
        raise PredictionFailedException(str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Batch Diagnostic Analysis on Multiple Images",
    description=(
        "Upload multiple medical scans to execute automated diagnosis, "
        "explainability, and anomaly detection across all images."
    ),
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of medical scan images"),
    modality: Optional[str] = Form(None, description="Optional modality override for all images"),
    engine: MedicalAIEngine = Depends(get_engine),
    _auth: Optional[str] = Depends(verify_api_key),
) -> BatchPredictionResponse:
    if not files:
        raise InvalidImageException("At least one image file must be uploaded for batch prediction.")

    if modality and modality.lower() not in SUPPORTED_MODALITIES and modality != "Auto Detect":
        raise UnsupportedModalityException(modality)

    start_batch = time.perf_counter()
    results: List[PredictionResponse] = []
    errors: List[Dict[str, Any]] = []
    override = None if (not modality or modality == "Auto Detect") else modality.lower()

    for f in files:
        tmp_path = None
        try:
            tmp_path = _validate_and_sanitize_image_file(f)
            ctx: EngineContext = engine.execute(
                image_source=tmp_path,
                manual_modality_override=override,
            )
            resp = _build_prediction_response(ctx)
            _store_record_in_db(resp)
            results.append(resp)
        except Exception as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    total_ms = (time.perf_counter() - start_batch) * 1000.0

    return BatchPredictionResponse(
        total_images=len(files),
        successful=len(results),
        failed=len(errors),
        total_ms=round(total_ms, 2),
        results=results,
        errors=errors,
    )
