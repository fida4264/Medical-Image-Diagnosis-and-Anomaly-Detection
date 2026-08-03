"""
execution_manager.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Safely executes registered strategies, unknown detector, and decision fusion.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from medical_ai_engine.anomaly.anomaly_result import AnomalyResult
from medical_ai_engine.anomaly.decision import DecisionEngine
from medical_ai_engine.anomaly.strategies import (
    ConfidenceThresholdStrategy,
    MahalanobisOODStrategy,
    StrategyConfiguration,
)
from medical_ai_engine.anomaly.unknown import UnknownDiseaseDetector
from .integration_configuration import IntegrationConfiguration
from .integration_exceptions import ExecutionManagerError
from .integration_metrics import AnomalyTimings

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Safely executes:
    1. Registered detection strategies (Confidence threshold, Mahalanobis OOD)
    2. Unknown Disease Detector
    3. Master Decision Fusion Engine

    Thread-Safety
    -------------
    Safe for concurrent execution across requests. Each request gets its own
    AnomalyContext, so no shared mutable state between concurrent calls.
    """

    def __init__(
        self,
        config: Optional[IntegrationConfiguration] = None,
        registry: Optional[AnomalyRegistry] = None,
        unknown_detector: Optional[UnknownDiseaseDetector] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ) -> None:
        self._config = config or IntegrationConfiguration.default()
        self._lock = threading.Lock()

        # Initialize registry & register default strategies
        self._registry = registry or AnomalyRegistry()
        if not self._registry.list_strategies():
            strat_cfg = StrategyConfiguration()
            self._registry.register(ConfidenceThresholdStrategy(strat_cfg))
            self._registry.register(MahalanobisOODStrategy(strat_cfg))
            self._registry.initialize_all({})

        # Initialize Unknown Disease Detector & Decision Engine
        self._unknown_detector = unknown_detector or UnknownDiseaseDetector()
        self._decision_engine = decision_engine or DecisionEngine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, prediction_result: Any, engine_ctx: Any) -> AnomalyContext:
        """
        Build a fresh AnomalyContext from any prediction result / engine context.

        Resolves modality, confidence, request_id, session_id via getattr with
        safe fallbacks so the ExecutionManager works with real PredictionResult
        objects from Module 14 as well as lightweight test doubles.
        """
        request_id = (
            getattr(prediction_result, "request_id", None)
            or getattr(engine_ctx, "request_id", None)
            or "unknown"
        )
        session_id = (
            getattr(prediction_result, "session_id", None)
            or getattr(engine_ctx, "session_id", None)
            or "unknown"
        )
        return AnomalyContext(
            request_id=request_id,
            session_id=session_id,
            modality=getattr(prediction_result, "modality", "unknown"),
            predicted_class=getattr(prediction_result, "predicted_disease", "unknown"),
            confidence=float(getattr(prediction_result, "confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_pipeline(
        self,
        prediction_result: Any,
        engine_ctx: Any,
    ) -> Tuple[AnomalyResult, AnomalyTimings]:
        """
        Execute the full anomaly pipeline and return the final AnomalyResult and timing metrics.

        Parameters
        ----------
        prediction_result : Any
            PredictionResult from Module 14 (or any duck-typed equivalent).
        engine_ctx : Any
            Engine execution context.

        Returns
        -------
        Tuple[AnomalyResult, AnomalyTimings]
        """
        t_total_0 = time.perf_counter()
        t_strat_ms = 0.0
        t_unk_ms = 0.0
        t_fuse_ms = 0.0

        try:
            # Build a per-request AnomalyContext — no shared mutable state
            anomaly_ctx = self._build_context(prediction_result, engine_ctx)

            # ── Stage 1: Strategy execution (Confidence & OOD) ──────────────────
            t0 = time.perf_counter()
            confidence_sig: Dict[str, Any] = {}
            ood_sig: Dict[str, Any] = {}

            try:
                conf_strategy = self._registry.get_strategy("confidence_threshold")
                conf_strategy.detect(anomaly_ctx)
                confidence_sig = dict(anomaly_ctx.confidence_analysis or {})
                confidence_sig["is_low_confidence"] = confidence_sig.get(
                    "confidence_level", "HIGH"
                ) in ("LOW", "REJECT")
            except Exception as exc:
                logger.warning("[ExecutionManager] Confidence strategy skipped: %s", exc)

            try:
                ood_strategy = self._registry.get_strategy("mahalanobis")
                ood_strategy.detect(anomaly_ctx)
                ood_sig = dict(anomaly_ctx.ood_analysis or {})
                ood_sig["ood_score"] = ood_sig.get("feature_distance", 0.0)
            except Exception as exc:
                logger.warning("[ExecutionManager] OOD strategy skipped: %s", exc)

            t_strat_ms = (time.perf_counter() - t0) * 1000.0

            # ── Stage 2: Unknown Disease Detection ──────────────────────────────
            t0 = time.perf_counter()
            unk_res = self._unknown_detector.analyze(anomaly_ctx)
            unk_sig = unk_res.unknown_disease_analysis or {}
            t_unk_ms = (time.perf_counter() - t0) * 1000.0

            # ── Stage 3: Decision Fusion ─────────────────────────────────────────
            t0 = time.perf_counter()
            final_result = self._decision_engine.evaluate_signals(
                prediction_result=anomaly_ctx,
                confidence_signal=confidence_sig,
                ood_signal=ood_sig,
                unknown_signal=unk_sig,
            )
            t_fuse_ms = (time.perf_counter() - t0) * 1000.0

            t_total_ms = (time.perf_counter() - t_total_0) * 1000.0

            timings = AnomalyTimings(
                strategy_execution_ms=t_strat_ms,
                unknown_detection_ms=t_unk_ms,
                decision_fusion_ms=t_fuse_ms,
                anomaly_total_latency_ms=t_total_ms,
                overall_integration_ms=t_total_ms,
            )

            logger.info(
                "[ExecutionManager] Pipeline completed in %.2fms (status=%s, risk=%s)",
                t_total_ms,
                final_result.status.value,
                final_result.risk_level.value,
            )
            return final_result, timings

        except Exception as exc:
            raise ExecutionManagerError(
                f"Execution manager pipeline failed: {exc}"
            ) from exc


__all__ = ["ExecutionManager"]
