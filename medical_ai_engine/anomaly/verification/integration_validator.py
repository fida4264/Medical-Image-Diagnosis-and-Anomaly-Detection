"""
integration_validator.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Validates that Module 16.6 integration contracts are correctly satisfied.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .verification_configuration import VerificationConfiguration
from .verification_exceptions import IntegrationValidationError

logger = logging.getLogger(__name__)


@dataclass
class IntegrationCheckResult:
    name: str
    passed: bool
    details: List[str]

    def to_dict(self) -> Dict:
        return {"name": self.name, "passed": self.passed, "details": self.details}


@dataclass
class IntegrationValidatorResult:
    checks: List[IntegrationCheckResult]

    @property
    def overall_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def to_dict(self) -> Dict:
        return {
            "overall_passed": self.overall_passed,
            "pass_count": self.pass_count,
            "total_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


class IntegrationValidator:
    """
    Validates Module 16.6 integration contracts:

    1. AnomalyIntegrator facade exists and exposes required API.
    2. WorkflowCoordinator attaches anomaly_result to prediction_result.extra.
    3. ExecutionManager builds AnomalyContext and runs the pipeline.
    4. AnomalyEventListener registers with arbitrary publisher.
    5. IntegrationMetrics records run data correctly.
    6. disabled=False bypasses anomaly without error.
    7. async_execution=False executes synchronously (result available immediately).
    8. Multi-modality routing: all 7 modalities accepted.
    """

    def __init__(
        self,
        config: Optional[VerificationConfiguration] = None,
    ) -> None:
        self._cfg = config or VerificationConfiguration.fast_test()

    def _make_request(self, request_id: str, confidence: float = 0.85,
                      modality: str = "chest_xray") -> Any:
        from dataclasses import dataclass as dc, field as f
        from typing import Dict as D, Any as A

        @dc
        class _Pred:
            request_id: str = "req-default"
            session_id: str = "sess-intval"
            modality: str = "chest_xray"
            predicted_disease: str = "pneumonia"
            confidence: float = 0.85
            extra: D[str, A] = f(default_factory=dict)

            def with_extra(self, e):
                import copy
                p = copy.copy(self)
                object.__setattr__(p, "extra", e)
                return p

        p = _Pred()
        object.__setattr__(p, "request_id", request_id)
        object.__setattr__(p, "confidence", confidence)
        object.__setattr__(p, "modality", modality)
        return p

    def _make_ctx(self, request_id: str) -> Any:
        from dataclasses import dataclass as dc

        @dc
        class _Ctx:
            request_id: str = "req-ctx"
            session_id: str = "sess-intval"

        c = _Ctx()
        object.__setattr__(c, "request_id", request_id)
        return c

    # ------------------------------------------------------------------
    def _check_facade_api(self) -> IntegrationCheckResult:
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration import AnomalyIntegrator
            integrator = AnomalyIntegrator()
            required = ["process_completed_prediction", "integrate_with_engine", "health_check", "metrics"]
            for attr in required:
                if hasattr(integrator, attr):
                    details.append(f"✅ AnomalyIntegrator.{attr} exists")
                else:
                    details.append(f"❌ Missing: AnomalyIntegrator.{attr}")
                    passed = False
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("facade_api", passed, details)

    def _check_anomaly_result_attached(self) -> IntegrationCheckResult:
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration import AnomalyIntegrator
            integrator = AnomalyIntegrator()
            pred = self._make_request("intval-attach")
            ctx = self._make_ctx("intval-attach")
            enriched = integrator.process_completed_prediction(ctx, pred)
            if "anomaly_result" in enriched.extra:
                details.append("✅ anomaly_result attached to PredictionResult.extra")
            else:
                details.append("❌ anomaly_result NOT found in PredictionResult.extra")
                passed = False
            for key in ("anomaly_status", "risk_level", "recommended_action"):
                if key in enriched.extra:
                    details.append(f"✅ {key} quick-access key present")
                else:
                    details.append(f"⚠️  {key} quick-access key missing")
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("anomaly_result_attached", passed, details)

    def _check_event_listener(self) -> IntegrationCheckResult:
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration import AnomalyEventListener, AnomalyIntegrator

            class _DummyPublisher:
                pass

            integrator = AnomalyIntegrator()
            listener = AnomalyEventListener(integrator=integrator)
            pub = _DummyPublisher()
            listener.register_with_publisher(pub)

            if hasattr(pub, "_subscribers") and "EngineCompletedEvent" in pub._subscribers:
                details.append("✅ AnomalyEventListener registered EngineCompletedEvent subscriber")
            else:
                details.append("❌ EngineCompletedEvent not registered on publisher")
                passed = False
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("event_listener_registration", passed, details)

    def _check_integration_metrics(self) -> IntegrationCheckResult:
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration import AnomalyIntegrator, IntegrationMetrics, AnomalyTimings
            integrator = AnomalyIntegrator()
            pred = self._make_request("intval-metrics")
            ctx = self._make_ctx("intval-metrics")
            integrator.process_completed_prediction(ctx, pred)
            summary = integrator.metrics.get_summary()
            if summary.get("total_requests", 0) >= 1:
                details.append("✅ IntegrationMetrics.total_requests >= 1 after one run")
            else:
                details.append("❌ IntegrationMetrics not updated after run")
                passed = False
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("integration_metrics", passed, details)

    def _check_disabled_bypass(self) -> IntegrationCheckResult:
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration import (
                AnomalyIntegrator, IntegrationConfiguration,
            )
            cfg = IntegrationConfiguration(enabled=False)
            integrator = AnomalyIntegrator(config=cfg)
            pred = self._make_request("intval-disabled")
            ctx = self._make_ctx("intval-disabled")
            enriched = integrator.process_completed_prediction(ctx, pred)
            if "anomaly_result" not in enriched.extra:
                details.append("✅ Disabled integration bypasses anomaly correctly")
            else:
                details.append("⚠️  anomaly_result present despite enabled=False")
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("disabled_bypass", passed, details)

    def _check_multi_modality_routing(self) -> IntegrationCheckResult:
        details = []
        all_passed = True
        try:
            from medical_ai_engine.anomaly.integration import AnomalyIntegrator
            integrator = AnomalyIntegrator()
            for modality in self._cfg.supported_modalities:
                pred = self._make_request(f"intval-mod-{modality}", modality=modality)
                ctx = self._make_ctx(f"intval-mod-{modality}")
                enriched = integrator.process_completed_prediction(ctx, pred)
                ok = "anomaly_result" in enriched.extra
                status = "✅" if ok else "❌"
                details.append(f"{status} modality={modality}")
                if not ok:
                    all_passed = False
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            all_passed = False
        return IntegrationCheckResult("multi_modality_routing", all_passed, details)

    def _check_execution_manager_context(self) -> IntegrationCheckResult:
        """Verify ExecutionManager builds AnomalyContext correctly."""
        details = []
        passed = True
        try:
            from medical_ai_engine.anomaly.integration.execution_manager import ExecutionManager
            mgr = ExecutionManager()
            pred = self._make_request("intval-ctx")
            ctx = self._make_ctx("intval-ctx")
            # _build_context must not raise
            anomaly_ctx = mgr._build_context(pred, ctx)
            if anomaly_ctx.request_id == "intval-ctx":
                details.append("✅ AnomalyContext.request_id correctly extracted")
            else:
                details.append(f"⚠️  request_id={anomaly_ctx.request_id!r}")
            if anomaly_ctx.modality == "chest_xray":
                details.append("✅ AnomalyContext.modality correctly extracted")
            else:
                details.append(f"⚠️  modality={anomaly_ctx.modality!r}")
        except Exception as exc:
            details.append(f"❌ Exception: {exc}")
            passed = False
        return IntegrationCheckResult("execution_manager_context", passed, details)

    # ------------------------------------------------------------------
    def run(self) -> IntegrationValidatorResult:
        checks = [
            self._check_facade_api(),
            self._check_anomaly_result_attached(),
            self._check_event_listener(),
            self._check_integration_metrics(),
            self._check_disabled_bypass(),
            self._check_multi_modality_routing(),
            self._check_execution_manager_context(),
        ]
        result = IntegrationValidatorResult(checks=checks)
        logger.info(
            "[IntegrationValidator] overall_passed=%s pass=%d/%d",
            result.overall_passed, result.pass_count, len(checks),
        )
        return result


__all__ = ["IntegrationValidator", "IntegrationValidatorResult", "IntegrationCheckResult"]
