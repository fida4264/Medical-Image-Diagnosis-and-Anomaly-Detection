"""
anomaly_integrator.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Central integration facade connecting MedicalAIEngine and the Anomaly Detection Subsystem.

Architecture
------------
MedicalAIEngine → [PredictionCompleted] → ExplainabilityIntegrator (15.6) → AnomalyIntegrator (16.6)
                                                                                  ↓
                                                                         WorkflowCoordinator
                                                                                  ↓
                                                                         ExecutionManager
                                                                                  ↓
                                                                        DecisionEngine / AnomalyResult
                                                                                  ↓
                                                                       Enriched PredictionResult
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from .event_listener import AnomalyEventListener
from .execution_manager import ExecutionManager
from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import AnomalyIntegrationError
from .integration_metrics import IntegrationMetrics
from .integration_validator import IntegrationValidator
from .workflow_coordinator import WorkflowCoordinator

logger = logging.getLogger(__name__)


class AnomalyIntegrator:
    """
    Central integration facade for Module 16.6.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        workflow_coordinator: Optional[WorkflowCoordinator] = None,
    ) -> None:
        self._config = config or IntegrationConfiguration.default()
        self._lock = threading.RLock()
        self._metrics = IntegrationMetrics()
        self._validator = IntegrationValidator(
            supported_modalities=self._config.supported_modalities,
        )
        self._exec_mgr = ExecutionManager(config=self._config)
        self._coordinator = workflow_coordinator or WorkflowCoordinator(
            config=self._config,
            execution_manager=self._exec_mgr,
            validator=self._validator,
            metrics=self._metrics,
        )
        self._listener = AnomalyEventListener(
            integrator=self,
            async_execution=self._config.async_execution,
        )
        logger.info("[AnomalyIntegrator] Initialized (enabled=%s).", self._config.enabled)

    def process_completed_prediction(
        self,
        engine_ctx: Any,
        prediction_result: Any,
    ) -> Any:
        """
        Process a prediction result after prediction (and optional explainability).

        Returns enriched PredictionResult with extra["anomaly_result"] attached.
        """
        with self._lock:
            return self._coordinator.coordinate(engine_ctx, prediction_result)

    def integrate_with_engine(self, engine: Any) -> None:
        """
        Integrate directly with MedicalAIEngine or EngineIntegrator instance.
        """
        try:
            publisher = getattr(engine, "event_publisher", None)
            if publisher is not None:
                self._listener.register_with_publisher(publisher)
                logger.info("[AnomalyIntegrator] Integrated with engine event publisher.")
        except Exception as exc:
            raise AnomalyIntegrationError(f"Failed to integrate with engine: {exc}") from exc

    def health_check(self) -> Dict[str, Any]:
        return {
            "component": "AnomalyIntegrator",
            "module": "16.6",
            "status": "healthy",
            "metrics": self._metrics.get_summary(),
        }

    @property
    def metrics(self) -> IntegrationMetrics:
        return self._metrics


__all__ = ["AnomalyIntegrator"]
