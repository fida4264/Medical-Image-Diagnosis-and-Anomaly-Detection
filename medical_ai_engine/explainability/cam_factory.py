"""
Medical AI Engine — CAM Factory

Module 15.1: Explainability Framework Architecture

Factory pattern implementation for instantiating CAM method plugins
from CAMRegistry.
"""

from __future__ import annotations

import logging
from typing import Optional

from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod
from medical_ai_engine.explainability.cam_registry import CAMRegistry

logger = logging.getLogger("CAMFactory")


class CAMFactory:
    """
    Factory for creating CAM method instances.
    """

    def __init__(self, registry: Optional[CAMRegistry] = None) -> None:
        self.registry = registry or CAMRegistry.get_instance()

    def create_cam_method(self, method_key: str) -> BaseCAMMethod:
        """
        Create and return an instance of the requested CAM method.

        Args:
            method_key: Registered method key (e.g. 'gradcam', 'scorecam').

        Returns:
            Instance of BaseCAMMethod subclass.

        Raises:
            KeyError: If method_key is not registered in CAMRegistry.
        """
        key = method_key.lower()
        if not self.registry.is_registered(key):
            available = ", ".join(self.registry.list_methods())
            raise KeyError(
                f"Unsupported CAM method '{method_key}'. Registered methods: [{available}]"
            )
        return self.registry.get_method_instance(key)

    @classmethod
    def create_default(cls) -> BaseCAMMethod:
        """Convenience method returning default 'gradcam' method."""
        return cls().create_cam_method("gradcam")
