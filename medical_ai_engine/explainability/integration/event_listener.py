"""
event_listener.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Event-driven bridge between EngineEventPublisher and ExplainabilityIntegrator.

Design
------
The existing EngineEventPublisher (Module 14.2.1) is a one-way publisher
with no subscriber registry. Module 15.6 works around this by:

  1. Offering a direct callback API (on_engine_completed) for synchronous
     integration via the WorkflowCoordinator.
  2. Offering a register_with_publisher() that monkey-patches a _subscribers
     dict into the EngineEventPublisher for lightweight event routing
     (zero modification to existing code — the publisher grows a new attribute
     only at runtime when the listener registers).

The listener always delegates to ExplainabilityIntegrator.process_completed_event()
and never performs any explainability work itself.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from .integration_exceptions import EventListenerError

logger = logging.getLogger(__name__)


class ExplainabilityEventListener:
    """
    Event-driven subscriber for MedicalAIEngine prediction-completed events.

    Two integration modes
    --------------------
    **Direct callback (recommended)**
        WorkflowCoordinator calls `on_engine_completed(engine_ctx, prediction_result)`
        synchronously or in a background thread after every successful prediction.

    **Publisher registration (optional)**
        Call `register_with_publisher(publisher)` to inject a lightweight
        subscription mechanism into the existing EngineEventPublisher without
        modifying its source.
    """

    def __init__(
        self,
        integrator: Any,               # ExplainabilityIntegrator (typed loosely)
        async_execution: bool = False,
    ) -> None:
        self._integrator = integrator
        self._async = async_execution
        self._lock = threading.Lock()
        self._active_threads: List[threading.Thread] = []

    # ── Primary Integration Hook ──────────────────────────────────────────────

    def on_engine_completed(
        self,
        engine_ctx: Any,
        prediction_result: Any,
    ) -> Optional[Any]:
        """
        Called after every successful prediction.

        Parameters
        ----------
        engine_ctx:
            EngineContext from MedicalAIEngine.execute().
        prediction_result:
            PredictionResult built by ResponseBuilder.

        Returns
        -------
        enriched PredictionResult if sync; None if async.
        """
        if self._async:
            t = threading.Thread(
                target=self._dispatch,
                args=(engine_ctx, prediction_result),
                daemon=True,
                name=f"xai-{getattr(engine_ctx, 'request_id', 'unknown')[:8]}",
            )
            with self._lock:
                self._active_threads.append(t)
            t.start()
            return None

        return self._dispatch(engine_ctx, prediction_result)

    # ── Publisher Registration ────────────────────────────────────────────────

    def register_with_publisher(self, publisher: Any) -> None:
        """
        Register this listener with an EngineEventPublisher for EngineCompleted events.

        Injects a lightweight `_subscribers` dict onto the publisher instance
        and monkey-patches the `publish` method to also dispatch to subscribers.

        Does NOT modify the publisher's source code.
        """
        try:
            if not hasattr(publisher, "_subscribers"):
                publisher._subscribers = {}
                original_publish = publisher.publish

                def _patched_publish(event: Any) -> None:
                    original_publish(event)
                    event_name = getattr(event, "event_name", "")
                    cbs: List[Callable] = publisher._subscribers.get(event_name, [])
                    for cb in cbs:
                        try:
                            cb(event)
                        except Exception as exc:
                            logger.warning(
                                "[ExplainabilityEventListener] Subscriber callback error: %s", exc
                            )

                publisher.publish = _patched_publish
                logger.info(
                    "[ExplainabilityEventListener] Monkey-patched EngineEventPublisher with subscriber support."
                )

            # Register for EngineCompleted
            publisher._subscribers.setdefault("EngineCompleted", []).append(
                self._on_engine_event
            )
            logger.info("[ExplainabilityEventListener] Registered for EngineCompleted events.")
        except Exception as exc:
            raise EventListenerError(
                f"Failed to register with EngineEventPublisher: {exc}"
            ) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_engine_event(self, event: Any) -> None:
        """Bridge EngineCompletedEvent → on_engine_completed (sync)."""
        try:
            # EngineCompletedEvent carries only request_id/session_id/duration.
            # Full engine_ctx and prediction_result are not available here —
            # this path is used when only publisher integration (not direct) is active.
            payload = {
                "request_id": getattr(event, "request_id", ""),
                "session_id": getattr(event, "session_id", ""),
                "duration_ms": getattr(event, "duration_ms", 0.0),
            }
            logger.debug(
                "[ExplainabilityEventListener] EngineCompleted event received: %s",
                payload.get("request_id"),
            )
            # When only event data is available (no full ctx), delegate to integrator
            # with a minimal payload dict
            if hasattr(self._integrator, "process_minimal_event"):
                self._integrator.process_minimal_event(payload)
        except Exception as exc:
            logger.warning("[ExplainabilityEventListener] _on_engine_event error: %s", exc)

    def _dispatch(self, engine_ctx: Any, prediction_result: Any) -> Any:
        """Call integrator and return enriched result."""
        try:
            return self._integrator.process_completed_event(engine_ctx, prediction_result)
        except Exception as exc:
            logger.error("[ExplainabilityEventListener] Dispatch error: %s", exc)
            return prediction_result

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def wait_for_pending(self, timeout: float = 30.0) -> None:
        """Block until all async explainability threads complete."""
        with self._lock:
            threads = list(self._active_threads)
        for t in threads:
            t.join(timeout=timeout)
        with self._lock:
            self._active_threads = [t for t in self._active_threads if t.is_alive()]


__all__ = ["ExplainabilityEventListener"]
