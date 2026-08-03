"""
Medical AI Engine — Explainability JSON Builder

Module 15.1: Explainability Framework Architecture

Serializes ExplainabilityResult objects into standardized JSON payloads.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from medical_ai_engine.explainability.explainability_result import ExplainabilityResult

logger = logging.getLogger("ExplainabilityJSONBuilder")


class ExplainabilityJSONBuilder:
    """
    Serializes ExplainabilityResult objects into JSON and dictionary representations.
    """

    @staticmethod
    def to_dict(result: ExplainabilityResult) -> Dict[str, Any]:
        """Convert ExplainabilityResult to dictionary."""
        return result.to_dict()

    @staticmethod
    def to_json(result: ExplainabilityResult, indent: bool = False) -> str:
        """Serialize ExplainabilityResult to JSON string."""
        if indent:
            return result.to_pretty_json()
        return result.to_json()
