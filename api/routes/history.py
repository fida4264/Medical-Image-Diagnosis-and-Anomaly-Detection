"""
api/routes/history.py — Module 21: Production REST API
History endpoints for searching, filtering, and retrieving stored predictions.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, Query, status

from api.dependencies import get_query_engine, verify_api_key
from api.exceptions import PredictionNotFoundException
from api.schemas import HistoryListResponse, HistoryRecordResponse
from medical_ai_engine.database.query_engine import QueryEngine

router = APIRouter(prefix="", tags=["Prediction History"])


@router.get(
    "/history",
    response_model=HistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Prediction History Audit Log",
    description="Search, filter, and paginate through saved diagnostic prediction records in the SQLite database.",
)
async def query_history(
    modality: Optional[str] = Query(None, description="Filter by imaging modality (e.g. 'chest_xray', 'brain_mri')"),
    disease: Optional[str] = Query(None, description="Filter by predicted disease name"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level: LOW, MODERATE, HIGH, CRITICAL"),
    anomaly_status: Optional[str] = Query(None, description="Filter by anomaly status: KNOWN_DISEASE or UNKNOWN_ANOMALY"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score threshold"),
    start_date: Optional[str] = Query(None, description="ISO start date filter (inclusive)"),
    end_date: Optional[str] = Query(None, description="ISO end date filter (inclusive)"),
    limit: int = Query(50, ge=1, le=200, description="Page size limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    order_by: str = Query("created_at", description="Sort column: created_at, confidence, modality, predicted_disease, risk_level, processing_time_ms"),
    order_dir: str = Query("DESC", description="Sort direction: ASC or DESC"),
    query_engine: QueryEngine = Depends(get_query_engine),
    _auth: Optional[str] = Depends(verify_api_key),
) -> HistoryListResponse:
    records = query_engine.search(
        modality=modality,
        disease=disease,
        risk_level=risk_level,
        anomaly_status=anomaly_status,
        patient_id=patient_id,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )

    record_responses = [HistoryRecordResponse(**r.to_dict()) for r in records]

    return HistoryListResponse(
        total_records=len(record_responses),
        limit=limit,
        offset=offset,
        records=record_responses,
    )


@router.get(
    "/history/{prediction_id}",
    response_model=HistoryRecordResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Single Prediction Record Details",
    description="Retrieve full details for a specific prediction ID stored in the database.",
)
async def get_prediction_by_id(
    prediction_id: str,
    query_engine: QueryEngine = Depends(get_query_engine),
    _auth: Optional[str] = Depends(verify_api_key),
) -> HistoryRecordResponse:
    rec = query_engine.get_by_id(prediction_id)
    if not rec:
        raise PredictionNotFoundException(prediction_id)
    return HistoryRecordResponse(**rec.to_dict())
