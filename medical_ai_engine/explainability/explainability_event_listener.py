"""
Medical AI Engine — Explainability Event Listener

Module 15.1: Explainability Framework Architecture

Subscribes to PredictionCompletedEvent via the existing EngineEventPublisher
event bus without introducing tight coupling to MedicalAIEngine.

Event Bus Design:
  - Subscribes to events published by medical_ai_engine.events.engine_events
  - Builds an ExplainabilityContext from the PredictionResult payload
  - Delegates to ExplainabilityEngine.explain()
  - Does NOT modify the original PredictionResult (immutable)
  - Fires asynchronously to avoid blocking the main prediction pipeline
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Optional

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration
from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.explainability_result import ExplainabilityResult

logger = logging.getLogger("ExplainabilityEventListener")


class ExplainabilityEventListener:
    """
    Event-driven adapter that subscribes to MedicalAIEngine prediction-completed
    events and triggers the Explainability Engine without tight coupling.

    Design Pattern: Observer / Event Subscriber

    Integration Point:
        The existing `EngineEventPublisher` in `medical_ai_engine.events.engine_events`
        provides a lightweight event bus. This listener registers a callback
        for 'EngineCompleted' events.

    Decoupling Strategy:
        - Does NOT import MedicalAIEngine directly.
        - Does NOT modify EngineContext (immutable contract).
        - Receives a plain dict payload built from PredictionResult.to_dict().
        - Constructs an independent ExplainabilityContext.
        - Publishes results to a registered result_callback (dependency injected).
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfiguration] = None,
        result_callback: Optional[Callable[[ExplainabilityResult], None]] = None,
        async_execution: bool = True,
    ) -> None:
        self.config = config or ExplainabilityConfiguration.default()
        self.result_callback = result_callback
        self.async_execution = async_execution
        self._active_threads: list[threading.Thread] = []

    def on_prediction_completed(self, prediction_payload: dict) -> None:
        """
        Callback invoked when an EngineCompleted event is fired.

        Args:
            prediction_payload: Dictionary representation of a PredictionResult.
        """
        if not self.config.enabled:
            return

        confidence = prediction_payload.get("confidence", 0.0)
        if confidence < self.config.min_confidence_for_explanation:
            logger.debug(
                "[ExplainabilityEventListener] Skipping explain — confidence %.2f < threshold %.2f",
                confidence,
                self.config.min_confidence_for_explanation,
            )
            return

        if self.async_execution:
            t = threading.Thread(
                target=self._run_explanation,
                args=(prediction_payload,),
                daemon=True,
                name=f"xai-{prediction_payload.get('request_id', 'unknown')[:8]}",
            )
            t.start()
            self._active_threads.append(t)
        else:
            self._run_explanation(prediction_payload)

    def _run_explanation(self, payload: dict) -> None:
        """Internal method executing the explainability workflow."""
        from medical_ai_engine.explainability.explainability_engine import ExplainabilityEngine

        request_id = payload.get("request_id", f"xai-{uuid.uuid4().hex[:8]}")
        session_id = payload.get("session_id", f"sess-{request_id}")

        ctx = ExplainabilityContext(
            request_id=request_id,
            session_id=session_id,
            modality=payload.get("modality", "unknown"),
            predicted_class=payload.get("predicted_disease", "unknown"),
            predicted_class_index=payload.get("predicted_class_index", 0),
            confidence=payload.get("confidence", 0.0),
            cam_method_key=self.config.default_cam_method,
        )

        try:
            engine = ExplainabilityEngine(config=self.config)
            result = engine.explain(ctx)
            logger.info(
                "[ExplainabilityEventListener] Explanation completed for '%s': status=%s, time=%.1fms",
                request_id,
                result.status,
                result.generation_time_ms,
            )
            if self.result_callback:
                self.result_callback(result)
        except Exception as e:
            logger.error("[ExplainabilityEventListener] Explanation failed for '%s': %s", request_id, e)

    def subscribe(self, event_publisher: Any) -> None:
        """
        Subscribe this listener to an EngineEventPublisher instance.

        Args:
            event_publisher: EngineEventPublisher from medical_ai_engine.events.engine_events.

        Example:
            listener = ExplainabilityEventListener()
            listener.subscribe(engine_integrator.engine.event_publisher)
        """
        try:
            event_publisher.subscribe("EngineCompleted", self._on_engine_event)
            logger.info("[ExplainabilityEventListener] Subscribed to EngineCompleted events.")
        except AttributeError:
            logger.warning(
                "[ExplainabilityEventListener] event_publisher.subscribe() not available. "
                "Manual integration via on_prediction_completed() required."
            )

    def _on_engine_event(self, event: Any) -> None:
        """Bridge between EngineEvent objects and prediction_payload dict."""
        try:
            payload = {
                "request_id": getattr(event, "request_id", ""),
                "session_id": getattr(event, "session_id", ""),
                "modality": getattr(event, "modality", "unknown"),
                "predicted_disease": getattr(event, "predicted_disease", "unknown"),
                "confidence": getattr(event, "confidence", 0.0),
            }
            self.on_prediction_completed(payload)
        except Exception as e:
            logger.warning("[ExplainabilityEventListener] Event bridge error: %s", e)
