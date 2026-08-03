"""
explainability_integrator.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Central integration facade connecting MedicalAIEngine and the XAI subsystem.

Architecture
------------
ExplainabilityIntegrator is the single integration point that:
  1. Wraps the existing WorkflowExecutor (Module 14.2.6)
  2. Attaches an ExplainabilityEventListener
  3. Dispatches post-prediction explainability via WorkflowCoordinator
  4. Returns enriched PredictionResult with gradcam_result + clinical explanation

It does NOT modify MedicalAIEngine, EngineContext, or any prediction module.
All explainability logic is isolated inside the explainability/ sub-package.

Usage
-----
::

    integrator = ExplainabilityIntegrator()
    result = integrator.execute_with_explanation(image, modality="brain_mri")
    print(result.gradcam_result["clinical_summary"])
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Union
from pathlib import Path

from .event_listener import ExplainabilityEventListener
from .execution_manager import ExecutionManager
from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import ExplainabilityIntegrationError
from .integration_metrics import IntegrationMetrics
from .integration_validator import IntegrationValidator
from .workflow_coordinator import WorkflowCoordinator

logger = logging.getLogger(__name__)


class ExplainabilityIntegrator:
    """
    Central integration facade for Module 15.6.

    Connects:
      MedicalAIEngine  →  WorkflowExecutor  →  ExplainabilityIntegrator
                                                        ↓
                                               WorkflowCoordinator
                                                        ↓
                                               ExecutionManager
                                                   ↙         ↘
                                   ExplainabilityPipeline   ClinicalExplanationEngine
                                         (15.1-15.4)            (15.5)
                                                        ↓
                                               ExplainabilityResult
                                                        ↓
                                          Enriched PredictionResult
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        engine_integrator: Optional[Any] = None,   # EngineIntegrator (Module 14.2.6)
    ) -> None:
        self._config = config or IntegrationConfiguration.fast_test()
        self._lock = threading.RLock()
        self._metrics = IntegrationMetrics()
        self._validator = IntegrationValidator(
            supported_modalities=self._config.supported_modalities,
        )
        self._exec_manager = ExecutionManager(config=self._config)
        self._coordinator = WorkflowCoordinator(
            config=self._config,
            execution_manager=self._exec_manager,
            validator=self._validator,
            metrics=self._metrics,
        )
        self._listener = ExplainabilityEventListener(
            integrator=self,
            async_execution=self._config.async_execution,
        )

        # Optionally wrap an EngineIntegrator
        self._engine_integrator = engine_integrator
        if engine_integrator is not None:
            self._attach_to_engine_integrator(engine_integrator)

        logger.info(
            "[ExplainabilityIntegrator] Initialized (enabled=%s async=%s cam=%s).",
            self._config.enabled,
            self._config.async_execution,
            self._config.cam_method,
        )

    # ── Primary Integration Entrypoint ────────────────────────────────────────

    def execute_with_explanation(
        self,
        image_source: Union[str, Path, Any, bytes],
        manual_modality_override: Optional[str] = None,
        request_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a full prediction + explainability workflow.

        This is the recommended entry point when using Module 15.6.
        It calls the platform prediction engine and then automatically
        dispatches explainability for every successful result.

        Parameters
        ----------
        image_source:
            Input image (file path, PIL Image, or bytes).
        manual_modality_override:
            Skip modality detection and route directly.
        request_id:
            Optional UUID for the request.
        extra_metadata:
            Custom key-value metadata.

        Returns
        -------
        Enriched PredictionResult with gradcam_result populated.
        """
        if self._engine_integrator is None:
            # Lazy initialization of EngineIntegrator
            from medical_ai_engine.integration.engine_integrator import EngineIntegrator
            self._engine_integrator = EngineIntegrator()
            self._attach_to_engine_integrator(self._engine_integrator)

        from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
        executor = WorkflowExecutor(integrator=self._engine_integrator)

        # Step 1: Run prediction pipeline (Modules 14.x)
        prediction_result, _, engine_ctx = executor.execute(
            image_source=image_source,
            manual_modality_override=manual_modality_override,
            request_id=request_id,
            extra_metadata=extra_metadata,
            format="dict",
        )

        # Step 2: Dispatch explainability (Modules 15.1–15.5)
        if self._config.enabled and prediction_result.is_success:
            prediction_result = self._coordinator.coordinate(engine_ctx, prediction_result)

        return prediction_result

    # ── Event Callback (called by ExplainabilityEventListener) ───────────────

    def process_completed_event(
        self,
        engine_ctx: Any,
        prediction_result: Any,
    ) -> Any:
        """
        Process a completed prediction event and attach explainability.
        Called directly by ExplainabilityEventListener.on_engine_completed().
        """
        if not self._config.enabled:
            return prediction_result
        return self._coordinator.coordinate(engine_ctx, prediction_result)

    def process_minimal_event(self, payload: Dict[str, Any]) -> None:
        """
        Handle minimal event payload from publisher subscription (no full ctx).
        Used when integration via publisher monkey-patch (no ctx available).
        Logs the event; full explainability requires execute_with_explanation().
        """
        logger.debug(
            "[ExplainabilityIntegrator] Minimal event received for request=%s",
            payload.get("request_id"),
        )

    # ── Direct Dispatch (for external callers) ────────────────────────────────

    def dispatch(self, engine_ctx: Any, prediction_result: Any) -> Any:
        """
        Directly dispatch explainability after an already-executed prediction.

        Use this when you have access to both the EngineContext and PredictionResult
        and want to attach explainability without going through execute_with_explanation().
        """
        return self._coordinator.coordinate(engine_ctx, prediction_result)

    # ── Listener Registration ─────────────────────────────────────────────────

    def register_listener_with_publisher(self, publisher: Any) -> None:
        """
        Register the event listener with an EngineEventPublisher.
        Enables automatic explainability on every EngineCompleted event.
        """
        self._listener.register_with_publisher(publisher)
        logger.info("[ExplainabilityIntegrator] Listener registered with EngineEventPublisher.")

    # ── Health & Metrics ──────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Return integration health status."""
        return {
            "enabled": self._config.enabled,
            "async_execution": self._config.async_execution,
            "cam_method": self._config.cam_method,
            "min_confidence_threshold": self._config.min_confidence_threshold,
            "has_engine_integrator": self._engine_integrator is not None,
            "metrics": self._metrics.snapshot(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.snapshot()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _attach_to_engine_integrator(self, engine_integrator: Any) -> None:
        """Optionally register listener with the engine's event publisher."""
        try:
            publisher = engine_integrator.engine.event_publisher
            self._listener.register_with_publisher(publisher)
        except Exception as exc:
            logger.warning(
                "[ExplainabilityIntegrator] Could not attach listener to publisher: %s", exc
            )


__all__ = ["ExplainabilityIntegrator"]
