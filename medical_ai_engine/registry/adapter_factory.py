"""
Medical AI Engine — Adapter Factory

Encapsulates creation of IPredictorAdapter instances.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
from medical_ai_engine.interfaces.adapter_interface import IAdapterFactory
from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter


class AdapterFactory(IAdapterFactory):
    """
    Factory for creating IPredictorAdapter objects.
    """

    def create_adapter(
        self,
        modality: str,
        adapter_cls: Type[IPredictorAdapter],
        config: Optional[Dict[str, Any]] = None,
    ) -> IPredictorAdapter:
        """
        Instantiate an uninitialized or initialized adapter instance.

        Args:
            modality: Target modality key string.
            adapter_cls: Adapter class implementing IPredictorAdapter.
            config: Optional configuration dictionary.

        Returns:
            IPredictorAdapter instance.
        """
        adapter = adapter_cls()
        adapter.initialize(config=config)
        return adapter
