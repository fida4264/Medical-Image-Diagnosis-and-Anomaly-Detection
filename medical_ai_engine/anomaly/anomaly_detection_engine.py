"""
anomaly_detection_engine.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Master AnomalyDetectionEngine orchestrator.

Data Flow
---------
  PredictionResult
      ↓
  AnomalyDetectionEngine.analyze()
      ↓
  [Step 1]  AnomalyContext construction
      ↓
  [Step 2]  ConfidenceAnalyzer
      ↓
  [Step 3]  OODDetector
      ↓
  [Step 4]  UnknownDiseaseDetector
      ↓
  [Step 5]  RiskAssessor
      ↓
  [Step 6]  DecisionEngine → AnomalyResult
      ↓
  PredictionResult.extra["anomaly_result"] ← attached

Failure Policy
--------------
- Any stage failure is caught; pipeline continues to next stage.
- PredictionResult is NEVER modified in a destructive way.
- AnomalyResult is always produced (status=NOT_ANALYZED on total failure).
- Primary prediction (class, confidence, explainability) is always preserved.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_exceptions import AnomalyDetectionError
from .anomaly_registry import AnomalyRegistry
from .anomaly_result import AnomalyResult, AnomalyStatus, RecommendedAction, RiskLevel
from .confidence_analyzer import ConfidenceAnalyzer
from .decision_engine import DecisionEngine
from .ood_detector import OODDetector
from .risk_assessor import RiskAssessor
from .unknown_disease_detector import UnknownDiseaseDetector

logger = logging.getLogger(__name__)


class AnomalyDetectionEngine:
    """
    Master orchestrator for the Anomaly Detection Framework (Module 16).

    Responsibilities
    ----------------
    1. Builds AnomalyContext from prediction inputs.
    2. Dispatches through the 5-stage detection pipeline.
    3. Attaches the immutable AnomalyResult to the PredictionResult.
    4. Guarantees fault-tolerant execution (prediction never breaks).

    Thread Safety
    -------------
    AnomalyDetectionEngine is stateless at the request level.
    All request state is encapsulated in AnomalyContext (per-call).
    All sub-components are individually thread-safe.
    Multiple concurrent requests are fully supported.

    Usage
    -----
    ::

        engine = AnomalyDetectionEngine()
        prediction_result = engine.analyze(prediction_result, engine_ctx=ctx)
        print(prediction_result.extra["anomaly_result"]["status"])
    """

    def __init__(
        self,
        config: Optional[AnomalyConfiguration] = None,
        registry: Optional[AnomalyRegistry] = None,
    ) -> None:
        self._config = config or AnomalyConfiguration.fast_test()
        self._registry = registry or AnomalyRegistry()

        # Initialize sub-components
        self._confidence_analyzer = ConfidenceAnalyzer(self._config)
        self._ood_detector = OODDetector(self._config, self._registry)
        self._unknown_detector = UnknownDiseaseDetector(self._config, self._registry)
        self._risk_assessor = RiskAssessor(self._config)
        self._decision_engine = DecisionEngine(self._config)

        self._total_analyzed: int = 0
        self._total_anomalies: int = 0
        self._lock = threading.Lock()

        logger.info(
            "[AnomalyDetectionEngine] Initialized (enabled=%s, strategy=%s).",
            self._config.enabled,
            self._config.default_strategy,
        )

    # ── Primary Entry Point ───────────────────────────────────────────────────

    def analyze(
        self,
        prediction_result: Any,
        engine_ctx: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Run the full anomaly detection pipeline on a completed prediction.

        Parameters
        ----------
        prediction_result : PredictionResult
            The prediction to analyze. Never modified destructively.
        engine_ctx : Optional[EngineContext]
            Optional engine context for feature vector access (Module 16.2+).
        request_id : Optional[str]
            Override request ID (auto-generated from prediction_result if None).

        Returns
        -------
        PredictionResult
            Enriched with prediction_result.extra["anomaly_result"] attached.
        """
        if not self._config.enabled:
            return prediction_result

        t_start = time.perf_counter()

        try:
            # Step 1: Build AnomalyContext from prediction inputs
            ctx = self._build_context(prediction_result, engine_ctx, request_id)
            ctx.execution_state = "RUNNING"
            ctx.set_timestamp("pipeline_start")

            # Step 2: Confidence Analysis
            if self._config.enable_confidence_analysis:
                self._confidence_analyzer.analyze(ctx)

            # Step 3: OOD Detection
            if self._config.enable_ood_detection:
                self._ood_detector.detect(ctx)

            # Step 4: Unknown Disease Detection
            if self._config.enable_unknown_disease_detection:
                self._unknown_detector.detect(ctx)

            # Step 5: Risk Assessment
            if self._config.enable_risk_assessment:
                self._risk_assessor.assess(ctx)

            # Step 6: Decision Engine → AnomalyResult
            total_ms = (time.perf_counter() - t_start) * 1000.0
            ctx.set_timestamp("pipeline_end")
            ctx.record_metric("total_anomaly_ms", total_ms)
            ctx.execution_state = "COMPLETED"

            anomaly_result = self._decision_engine.decide(ctx, total_ms)

            # Attach AnomalyResult to PredictionResult (non-destructive)
            enriched = self._attach_result(prediction_result, anomaly_result)

            # Update counters
            with self._lock:
                self._total_analyzed += 1
                if anomaly_result.is_anomalous:
                    self._total_anomalies += 1

            logger.info(
                "[AnomalyDetectionEngine] Analysis complete: request=%s status=%s "
                "risk=%s action=%s (%.2f ms)",
                ctx.request_id,
                anomaly_result.status.value,
                anomaly_result.risk_level.value,
                anomaly_result.recommended_action.value,
                total_ms,
            )

            return enriched

        except Exception as exc:
            # Absolute last-resort recovery: return original prediction untouched
            total_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception(
                "[AnomalyDetectionEngine] Fatal pipeline failure (%.2f ms): %s", total_ms, exc
            )
            return prediction_result

    # ── Context Construction ──────────────────────────────────────────────────

    def _build_context(
        self,
        prediction_result: Any,
        engine_ctx: Optional[Any],
        request_id: Optional[str],
    ) -> AnomalyContext:
        """Extract fields from PredictionResult into AnomalyContext."""
        rid = request_id or getattr(prediction_result, "request_id", None) or str(uuid.uuid4())
        sid = getattr(prediction_result, "session_id", rid)

        ctx = AnomalyContext(
            request_id=rid,
            session_id=sid,
            modality=getattr(prediction_result, "modality", "unknown"),
            predicted_class=getattr(prediction_result, "predicted_disease", "unknown"),
            confidence=float(getattr(prediction_result, "confidence", 0.0)),
            quality_score=None,  # populated from engine_ctx in Module 16.2+
        )

        # Extract quality score if available from EngineContext (Module 14/16.2+)
        if engine_ctx is not None:
            ctx.quality_score = getattr(engine_ctx, "quality_score", None)

        ctx.log("AnomalyContext constructed.", stage="AnomalyDetectionEngine")
        return ctx

    # ── Result Attachment ─────────────────────────────────────────────────────

    @staticmethod
    def _attach_result(prediction_result: Any, anomaly_result: AnomalyResult) -> Any:
        """
        Non-destructively attach AnomalyResult to PredictionResult.

        If PredictionResult is a frozen dataclass, use dataclasses.replace().
        Otherwise, write to .extra["anomaly_result"] dict field.
        """
        import dataclasses

        anomaly_dict = anomaly_result.to_dict()

        if dataclasses.is_dataclass(prediction_result):
            try:
                current_extra = dict(getattr(prediction_result, "extra", None) or {})
                current_extra["anomaly_result"] = anomaly_dict
                return dataclasses.replace(prediction_result, extra=current_extra)
            except Exception:
                # Fallback: try to set extra in-place (non-frozen dataclasses)
                try:
                    prediction_result.extra["anomaly_result"] = anomaly_dict
                except Exception:
                    pass
                return prediction_result

        # Plain dict or object
        try:
            if isinstance(prediction_result, dict):
                prediction_result["anomaly_result"] = anomaly_dict
            else:
                prediction_result.extra = getattr(prediction_result, "extra", {})
                prediction_result.extra["anomaly_result"] = anomaly_dict
        except Exception:
            pass

        return prediction_result

    # ── Registry & Configuration Access ──────────────────────────────────────

    def get_registry(self) -> AnomalyRegistry:
        return self._registry

    def get_config(self) -> AnomalyConfiguration:
        return self._config

    # ── Health & Metrics ──────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "default_strategy": self._config.default_strategy,
            "registered_strategies": self._registry.list_strategies(),
            "total_analyzed": self._total_analyzed,
            "total_anomalies": self._total_anomalies,
            "strategy_health": self._registry.health_report(),
        }

    def __repr__(self) -> str:
        return (
            f"AnomalyDetectionEngine("
            f"enabled={self._config.enabled}, "
            f"strategies={self._registry.list_strategies()}, "
            f"analyzed={self._total_analyzed})"
        )


__all__ = ["AnomalyDetectionEngine"]
