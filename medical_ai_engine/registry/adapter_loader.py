"""
Medical AI Engine — Adapter Loader

Responsible for instantiating adapter classes and triggering model weight loading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
from medical_ai_engine.interfaces.adapter_interface import IAdapterLoader
from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter
from medical_ai_engine.registry.registry_exceptions import AdapterLoadException


class AdapterLoader(IAdapterLoader):
    """
    Handles instantiating IPredictorAdapter classes and triggering load_model().
    """

    def load(
        self,
        adapter_cls: Type[IPredictorAdapter],
        checkpoint_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> IPredictorAdapter:
        """
        Instantiate adapter and load model weights.

        Args:
            adapter_cls: Class implementing IPredictorAdapter.
            checkpoint_path: Optional path to checkpoint file.
            label_map_path: Optional path to label map JSON file.
            config: Optional configuration dict.

        Returns:
            Fully loaded IPredictorAdapter instance.

        Raises:
            AdapterLoadException: If loading fails.
        """
        try:
            adapter = adapter_cls()
            adapter.initialize(config=config)
            adapter.load_model(checkpoint_path=checkpoint_path, label_map_path=label_map_path)
            return adapter
        except Exception as e:
            modality = getattr(adapter_cls, "modality", "unknown")
            if isinstance(modality, property):
                try:
                    modality = adapter_cls().modality
                except Exception:
                    modality = "unknown"
            raise AdapterLoadException(modality=str(modality), cause=e)
