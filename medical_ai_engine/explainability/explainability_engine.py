"""
Medical AI Engine — Explainability Engine Core

Module 15.1: Explainability Framework Architecture

Primary entry point for the Explainable AI (XAI) subsystem.
Implements IExplainabilityEngine.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Type

from medical_ai_engine.explainability.cam_factory import CAMFactory
from medical_ai_engine.explainability.cam_registry import CAMRegistry
from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration
from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.explainability_pipeline import ExplainabilityPipeline
from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
from medical_ai_engine.explainability.interfaces.explainability_engine_interface import IExplainabilityEngine
from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod

logger = logging.getLogger("ExplainabilityEngine")


class ExplainabilityEngine(IExplainabilityEngine):
    """
    Main orchestrator for Explainable AI (XAI) in the Medical AI Platform.

    Provides plugin-based CAM method registration, context execution, and
    event subscription.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfiguration] = None,
        registry: Optional[CAMRegistry] = None,
        pipeline: Optional[ExplainabilityPipeline] = None,
    ) -> None:
        self.config = config or ExplainabilityConfiguration.default()
        self._lock = threading.RLock()

        self.registry = registry or CAMRegistry.get_instance()
        self.factory = CAMFactory(registry=self.registry)
        self.pipeline = pipeline or ExplainabilityPipeline(
            config=self.config,
            cam_factory=self.factory,
        )

        logger.info(
            "[ExplainabilityEngine] Initialized. Default method: '%s', Registered methods: %s",
            self.config.default_cam_method,
            self.registry.list_methods(),
        )

    def explain(self, context: ExplainabilityContext) -> ExplainabilityResult:
        """
        Execute the explainability pipeline for a given ExplainabilityContext.

        Args:
            context: Populated ExplainabilityContext.

        Returns:
            Immutable ExplainabilityResult object.
        """
        with self._lock:
            if not self.config.enabled:
                logger.info("[ExplainabilityEngine] Explainability engine disabled in configuration.")
                return ExplainabilityResult(
                    request_id=context.request_id,
                    session_id=context.session_id,
                    status="SKIPPED",
                    clinical_summary="Explainability engine is disabled in configuration.",
                )

            # Default CAM method if not set
            if not context.cam_method_key:
                context.cam_method_key = self.config.default_cam_method

            return self.pipeline.execute(context)

    def register_cam_method(self, method_key: str, method_cls: Type[BaseCAMMethod]) -> None:
        """Register a new CAM method plugin dynamically."""
        with self._lock:
            self.registry.register(method_key, method_cls, overwrite=True)
            logger.info("[ExplainabilityEngine] Registered new CAM method plugin: '%s'", method_key)

    def get_supported_methods(self) -> Dict[str, str]:
        """Return a dictionary of registered CAM method keys to descriptions."""
        with self._lock:
            meta = self.registry.get_all_metadata()
            return {k: v.get("description", k) for k, v in meta.items()}

    def __repr__(self) -> str:
        return (
            f"ExplainabilityEngine("
            f"enabled={self.config.enabled}, "
            f"default_method={self.config.default_cam_method!r}, "
            f"methods_count={len(self.registry.list_methods())})"
        )
