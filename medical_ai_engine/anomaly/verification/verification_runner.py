"""
verification_runner.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Facade for executing the anomaly verification suite.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .production_validator import ProductionValidatorRunner
from .verification_configuration import VerificationConfiguration

logger = logging.getLogger(__name__)


class VerificationRunner:
    """
    Central VerificationRunner facade for Module 16.7.
    """

    def __init__(
        self,
        config: Optional[VerificationConfiguration] = None,
        integrator: Optional[Any] = None,
    ) -> None:
        self._cfg = config or VerificationConfiguration.fast_test()
        self._runner = ProductionValidatorRunner(config=self._cfg, integrator=integrator)

    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification, profiling, benchmarking, and stress tests."""
        return self._runner.run_all()


__all__ = ["VerificationRunner"]
