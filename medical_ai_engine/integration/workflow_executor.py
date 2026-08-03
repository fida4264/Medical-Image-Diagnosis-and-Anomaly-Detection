"""
Medical AI Engine — Workflow Executor

Executes complete end-to-end prediction workflows:
  Image Input
      ↓
  EngineContext Creation
      ↓
  Pipeline Execution (9 Stages)
      ↓
  Modality Classification & Routing
      ↓
  Adapter Disease Prediction
      ↓
  Confidence & Clinical Report Hooks
      ↓
  ResponseBuilder (Immutable PredictionResult)
      ↓
  ResponseValidator & Serializer
      ↓
  Performance Metrics Collection
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.exceptions import EngineExecutionException
from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
from medical_ai_engine.integration.integration_exceptions import (
    WorkflowExecutionException,
    IntegrationValidationException,
)
from medical_ai_engine.integration.integration_metrics import IntegrationMetrics
from medical_ai_engine.integration.integration_validator import IntegrationValidator
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_builder import ResponseBuilder
from medical_ai_engine.response.response_serializer import ResponseSerializer
from medical_ai_engine.response.response_schema import PredictionStatus

if TYPE_CHECKING:
    from medical_ai_engine.integration.engine_integrator import EngineIntegrator

logger = logging.getLogger("WorkflowExecutor")


class WorkflowExecutor:
    """
    Coordinates end-to-end prediction workflow execution across integrated subsystems.
    """

    def __init__(
        self,
        integrator: Optional["EngineIntegrator"] = None,
        config: Optional[IntegrationConfiguration] = None,
        metrics: Optional[IntegrationMetrics] = None,
    ) -> None:
        if integrator is not None:
            self.integrator = integrator
            self.config = integrator.config
        else:
            from medical_ai_engine.integration.engine_integrator import EngineIntegrator
            self.integrator = EngineIntegrator(config=config)
            self.config = self.integrator.config

        self.validator = IntegrationValidator(config=self.config)
        self.serializer = ResponseSerializer()
        self.metrics = metrics or IntegrationMetrics()

    def execute(
        self,
        image_source: Union[str, Path, Any, bytes],
        manual_modality_override: Optional[str] = None,
        request_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        format: str = "dict",
        expected_modality: Optional[str] = None,
    ) -> Tuple[PredictionResult, Union[str, Dict[str, Any]], EngineContext]:
        """
        Execute an end-to-end prediction workflow.

        Args:
            image_source: Input image (file path, PIL Image, or raw bytes).
            manual_modality_override: Optional target modality override.
            request_id: Optional UUID string.
            extra_metadata: Custom metadata dict.
            format: Output serialization format ('json', 'pretty_json', 'dict').
            expected_modality: Expected modality string to validate against.

        Returns:
            Tuple of (PredictionResult, SerializedResponse, EngineContext).
        """
        start_time = time.perf_counter()
        response_builder = self.integrator.response_builder

        try:
            # 1. Execute Engine Pipeline (Module 14.2.1 Core + Module 14.2.4 Pipeline)
            ctx = self.integrator.engine.execute(
                image_source=image_source,
                manual_modality_override=manual_modality_override,
                request_id=request_id,
                extra_metadata=extra_metadata,
            )

            # 2. Build Immutable PredictionResult (Module 14.2.5 Response Builder)
            resp_start = time.perf_counter()
            result = response_builder.build(ctx)
            resp_latency = (time.perf_counter() - resp_start) * 1000.0

            # 3. Validate End-to-End Integration
            self.validator.validate_workflow_execution(
                ctx=ctx,
                result=result,
                expected_modality=expected_modality,
            )

            # 4. Serialize Response
            serialized = self.serializer.serialize(result, fmt=format)

            # 5. Record Metrics
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
            routing_ms = ctx.performance_metrics.get("routing_ms", 0.0)
            adapter_ms = ctx.performance_metrics.get("inference_ms", 0.0)

            self.metrics.record_workflow(
                total_time_ms=total_time_ms,
                routing_latency_ms=routing_ms,
                adapter_latency_ms=adapter_ms,
                response_latency_ms=resp_latency,
                modality=result.modality,
                success=result.is_success,
            )

            return result, serialized, ctx

        except Exception as exc:
            total_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning("[WorkflowExecutor] Workflow failed: %s", exc)

            # Construct fallback error PredictionResult cleanly
            dummy_ctx = EngineContext(
                request_id=request_id or "unknown-request",
                session_id="error-session",
            )
            error_result = response_builder.build_error_response(
                ctx=dummy_ctx,
                error_message=str(exc),
            )
            serialized = self.serializer.serialize(error_result, fmt=format)

            self.metrics.record_workflow(
                total_time_ms=total_time_ms,
                modality="unknown",
                success=False,
            )

            if not self.config.catch_pipeline_exceptions:
                raise WorkflowExecutionException(
                    message=f"Prediction workflow failed: {exc}",
                    request_id=request_id,
                    cause=exc,
                ) from exc

            return error_result, serialized, dummy_ctx
