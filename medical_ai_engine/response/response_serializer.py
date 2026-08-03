"""
Medical AI Engine — Response Serializer

Handles serialization of PredictionResult to supported output formats.
Supported: 'json', 'pretty_json', 'dict'.
Reserved (not implemented): 'fhir_r4'.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Union, TYPE_CHECKING

from medical_ai_engine.response.response_exceptions import ResponseSerializationException
from medical_ai_engine.response.response_schema import SERIALIZATION_FORMATS

if TYPE_CHECKING:
    from medical_ai_engine.response.prediction_result import PredictionResult

logger = logging.getLogger("ResponseSerializer")


class ResponseSerializer:
    """
    Serializes a PredictionResult to the requested output format.
    """

    def serialize(
        self,
        result: "PredictionResult",
        fmt: str = "json",
    ) -> Union[str, Dict[str, Any]]:
        """
        Serialize PredictionResult to the specified format.

        Args:
            result: Immutable PredictionResult to serialize.
            fmt: Output format — one of 'json', 'pretty_json', 'dict'.

        Returns:
            Serialized output (str for json/pretty_json, dict for dict).

        Raises:
            ResponseSerializationException: On serialization failure or unsupported format.
        """
        if fmt not in SERIALIZATION_FORMATS:
            raise ResponseSerializationException(
                message=f"Unsupported serialization format '{fmt}'.",
                format_requested=fmt,
            )

        try:
            if fmt == "json":
                return result.to_json()
            elif fmt == "pretty_json":
                return result.to_pretty_json()
            elif fmt == "dict":
                return result.to_dict()
        except Exception as exc:
            raise ResponseSerializationException(
                message=f"Serialization failed for format '{fmt}'.",
                format_requested=fmt,
                cause=exc,
            ) from exc

    def serialize_to_json(self, result: "PredictionResult") -> str:
        """Convenience: serialize to compact JSON string."""
        return self.serialize(result, fmt="json")

    def serialize_to_pretty_json(self, result: "PredictionResult") -> str:
        """Convenience: serialize to indented JSON string."""
        return self.serialize(result, fmt="pretty_json")

    def serialize_to_dict(self, result: "PredictionResult") -> Dict[str, Any]:
        """Convenience: serialize to dictionary."""
        return self.serialize(result, fmt="dict")

    @staticmethod
    def deserialize_json(json_str: str) -> Dict[str, Any]:
        """
        Parse a JSON string back to a dict for testing/inspection.
        Does not reconstruct PredictionResult (frozen dataclass).
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ResponseSerializationException(
                message="Failed to deserialize JSON response.",
                format_requested="json",
                cause=exc,
            ) from exc
