"""
fusion_engine.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Fuses multiple anomaly detection signals into a unified decision output.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from .decision_configuration import DecisionConfiguration
from .decision_exceptions import FusionEngineError
from .decision_policy import DecisionOutput, IDecisionPolicy, PolicyRegistry

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Fuses multiple input anomaly detection signals:
    - Confidence threshold analysis
    - Mahalanobis distance OOD detection
    - Unknown disease detection
    - Image quality assessment

    Uses a configurable IDecisionPolicy to output a single DecisionOutput.

    Thread-Safety
    -------------
    Public methods are thread-safe and stateless per request.
    """

    def __init__(
        self,
        config: Optional[DecisionConfiguration] = None,
        policy: Optional[IDecisionPolicy] = None,
    ) -> None:
        self._cfg = config or DecisionConfiguration.default()
        self._registry = PolicyRegistry()
        if policy is not None:
            self._policy = policy
        else:
            self._policy = self._registry.get(self._cfg.policy_name, config=self._cfg)
        self._lock = threading.Lock()
        logger.info("FusionEngine initialised (policy=%s)", self._cfg.policy_name)

    def fuse(
        self,
        confidence_signal: Optional[Dict[str, Any]] = None,
        ood_signal: Optional[Dict[str, Any]] = None,
        unknown_signal: Optional[Dict[str, Any]] = None,
        quality_signal: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        modality: str = "chest_xray",
    ) -> DecisionOutput:
        """
        Fuse all input detection signals into a single DecisionOutput.

        Parameters
        ----------
        confidence_signal : Dict[str, Any] or None
            Output from confidence threshold analyzer strategy.
        ood_signal : Dict[str, Any] or None
            Output from Mahalanobis OOD strategy.
        unknown_signal : Dict[str, Any] or None
            Output from Unknown Disease Detector.
        quality_signal : Dict[str, Any] or None
            Output from image quality check.
        confidence : float
            Raw model prediction confidence.
        modality : str
            Medical imaging modality string.

        Returns
        -------
        DecisionOutput
        """
        t0 = time.perf_counter()
        try:
            signals = {
                "low_confidence": confidence_signal or {},
                "ood": ood_signal or {},
                "unknown_disease": unknown_signal or {},
                "poor_quality": quality_signal or {},
            }

            output = self._policy.evaluate(signals, confidence, modality)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "Fusion completed in %.2fms: status=%s risk=%s action=%s",
                elapsed_ms, output.status.value, output.risk_level.value, output.recommended_action.value,
            )
            return output

        except Exception as exc:
            raise FusionEngineError(f"Decision signal fusion failed: {exc}") from exc

    def set_policy(self, policy_name: str) -> None:
        """Switch active policy dynamically."""
        with self._lock:
            self._policy = self._registry.get(policy_name, config=self._cfg)
            self._cfg.policy_name = policy_name
            logger.info("Switched FusionEngine policy to '%s'", policy_name)

    @property
    def current_policy_name(self) -> str:
        return self._cfg.policy_name


__all__ = ["FusionEngine"]
