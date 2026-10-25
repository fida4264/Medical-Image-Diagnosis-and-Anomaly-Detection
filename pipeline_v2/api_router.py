"""
pipeline_v2/api_router.py
=========================
FastAPI routes for Phase 9 V2 Hardened Inference Pipeline.
Exposes endpoints for:
- GET  /api/v2/health        : Health, ready status, checkpoint availability
- GET  /api/v2/capabilities  : Modality configurations, classes, and frozen thresholds
- POST /api/v2/predict       : Hardened single-image inference with quality gates
- GET  /api/v2/audit         : Privacy-conscious provenance and audit log retrieval
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from pipeline_v2.config import (
    LEGAL_LIMITATIONS_NOTICE,
    MODALITY_SETTINGS,
    PIPELINE_VERSION,
    SUPPORTED_MODALITIES,
)
from pipeline_v2.inference_pipeline import get_pipeline_v2
from pipeline_v2.schemas import QualityGateState, StructuredInferenceResponse

router = APIRouter(prefix="/api/v2", tags=["V2 Hardened Pipeline"])


@router.get("/health", summary="Pipeline Health & Readiness")
async def get_health() -> Dict[str, Any]:
    """Returns pipeline readiness and checkpoint health across all 7 modalities."""
    pipeline = get_pipeline_v2()
    modality_health = {}
    all_ready = True

    for mod in SUPPORTED_MODALITIES:
        cfg = MODALITY_SETTINGS[mod]
        ckpt_exists = cfg["checkpoint"].exists()
        if not ckpt_exists:
            all_ready = False
        modality_health[mod] = {
            "task": cfg["task"],
            "classes": cfg["classes"],
            "checkpoint_available": ckpt_exists,
            "status": "READY" if ckpt_exists else "MISSING_CHECKPOINT",
        }

    return {
        "status": "HEALTHY" if all_ready else "DEGRADED",
        "pipeline_version": PIPELINE_VERSION,
        "ready": all_ready,
        "modalities": modality_health,
        "disclaimer": LEGAL_LIMITATIONS_NOTICE,
    }


@router.get("/capabilities", summary="Pipeline Capabilities & Frozen Thresholds")
async def get_capabilities() -> Dict[str, Any]:
    """Returns supported modalities, class maps, calibration info, and frozen OOD thresholds."""
    pipeline = get_pipeline_v2()
    return {
        "pipeline_version": PIPELINE_VERSION,
        "supported_modalities": SUPPORTED_MODALITIES,
        "modality_settings": {
            m: {
                "display_name": c["display_name"],
                "task": c["task"],
                "num_classes": c["num_classes"],
                "classes": c["classes"],
                "normal_reference": c["normal_class"],
                "cv_limitation": c.get("cv_limitation"),
            }
            for m, c in MODALITY_SETTINGS.items()
        },
        "supported_ood_methods": ["knn", "mahalanobis", "msp", "entropy", "energy"],
        "primary_ood_method": pipeline.primary_ood_method,
        "quality_gate_states": [s.value for s in QualityGateState],
        "disclaimer": LEGAL_LIMITATIONS_NOTICE,
    }


@router.post(
    "/predict",
    response_model=StructuredInferenceResponse,
    summary="Hardened Inference with Quality Gates",
)
async def predict_image(
    file: UploadFile = File(..., description="Medical imaging scan file"),
    modality: str = Form(..., description="Target imaging modality (one of 7 supported)"),
    patient_id: Optional[str] = Form(None, description="Optional patient pseudonym for audit tracking"),
    ood_method: Optional[str] = Form("knn", description="OOD detection method override (default: knn)"),
) -> StructuredInferenceResponse:
    """
    Executes the Phase 9 hardened inference pipeline:
    1. Multi-stage input validation & sanitization
    2. Modality configuration verification
    3. Baseline ResNet-18 forward pass
    4. Phase 7 calibration (or Retina per-label thresholding)
    5. Phase 8 OOD scoring against frozen validation threshold
    6. Phase 8 normal reference anomaly scoring
    7. Multi-stage quality-gate decision (accepts, flags, or abstains)
    8. Privacy-conscious audit logging
    """
    pipeline = get_pipeline_v2()

    # Read uploaded bytes
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file payload: {e}",
        )

    # Execute pipeline
    response = pipeline.predict(
        image_input=raw_bytes,
        modality=modality,
        patient_id=patient_id,
        ood_method_override=ood_method,
    )

    return response


@router.get("/audit", summary="Recent Privacy-Preserved Audit Records")
async def get_audit_records(limit: int = 50) -> Dict[str, Any]:
    """Retrieves recent inference audit records without raw pixel data or PHI."""
    pipeline = get_pipeline_v2()
    records = pipeline.audit_logger.get_recent_records(limit=limit)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "record_count": len(records),
        "records": records,
    }
