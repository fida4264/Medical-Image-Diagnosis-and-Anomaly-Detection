"""
event_listener.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Event listener subscribing to event publisher for EngineCompleted events.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from .integration_exceptions import EventListenerError

logger = logging.getLogger(__name__)


class AnomalyEventListener:
    """
    Event-driven subscriber for MedicalAIEngine events.

    Listens for EngineCompleted events and triggers post-prediction anomaly detection.
    """

    def __init__(self, integrator: Any, async_execution: bool = False) -> None:
        self._integrator = integrator
        self._async = async_execution
        self._lock = threading.Lock()
        self._active_threads: List[threading.Thread] = []

    def on_engine_completed(self, engine_ctx: Any, prediction_result: Any) -> Optional[Any]:
        """
        Called when engine completed prediction (and optional explainability).
        """
        if self._async:
            t = threading.Thread(
                target=self._dispatch,
                args=(engine_ctx, prediction_result),
                daemon=True,
                name=f"anomaly-{getattr(engine_ctx, 'request_id', 'req')[:8]}",
            )
            with self._lock:
                self._active_threads.append(t)
            t.start()
            return None

        return self._dispatch(engine_ctx, prediction_result)

    def _dispatch(self, engine_ctx: Any, prediction_result: Any) -> Any:
        try:
            return self._integrator.process_completed_prediction(engine_ctx, prediction_result)
        except Exception as exc:
            logger.exception("[AnomalyEventListener] Failed to dispatch anomaly event: %s", exc)
            return prediction_result

    def register_with_publisher(self, publisher: Any) -> None:
        """
        Register subscriber with EngineEventPublisher.
        Monkey-patches subscriber support if publisher lacks `_subscribers`.
        """
        try:
            if not hasattr(publisher, "_subscribers"):
                publisher._subscribers = {}
                original_publish = getattr(publisher, "publish", None)

                def patched_publish(event: Any) -> None:
                    if original_publish:
                        original_publish(event)
                    event_name = type(event).__name__
                    cbs: List[Callable] = publisher._subscribers.get(event_name, [])
                    for cb in cbs:
                        try:
                            cb(event)
                        except Exception as e:
                            logger.error("[AnomalyEventListener] Callback error: %s", e)

                publisher.publish = patched_publish

            publisher._subscribers.setdefault("EngineCompletedEvent", []).append(
                lambda evt: self.on_engine_completed(
                    getattr(evt, "engine_context", None),
                    getattr(evt, "prediction_result", None),
                )
            )
            logger.info("[AnomalyEventListener] Registered subscriber with EngineEventPublisher.")
        except Exception as exc:
            raise EventListenerError(f"Failed to register with publisher: {exc}") from exc


__all__ = ["AnomalyEventListener"]
