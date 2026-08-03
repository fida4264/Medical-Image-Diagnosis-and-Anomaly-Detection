"""
Medical AI Engine — ModelRegistry

The primary entry point for model discovery, registration, lazy loading,
caching, reloading, hot-swapping, and health assessment.
Implements IModelRegistry interface.

Thread-safe and modality-independent: the Engine Core interacts ONLY with ModelRegistry
and receives IPredictorAdapter instances without direct model imports.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Type

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter
from medical_ai_engine.interfaces.registry_interface import IModelRegistry
from medical_ai_engine.registry.adapter_cache import AdapterCache
from medical_ai_engine.registry.adapter_factory import AdapterFactory
from medical_ai_engine.registry.adapter_loader import AdapterLoader
from medical_ai_engine.registry.adapter_metadata import AdapterMetadata
from medical_ai_engine.registry.adapter_registry import AdapterRegistry
from medical_ai_engine.registry.registry_configuration import RegistryConfiguration
from medical_ai_engine.registry.registry_exceptions import (
    AdapterLoadException,
    AdapterNotFoundException,
    DuplicateAdapterRegistrationException,
)


class ModelRegistry(IModelRegistry):
    """
    Singleton / Managed Model Registry for the Medical AI Platform.

    Responsibilities:
      - Dynamic registration of IPredictorAdapter classes
      - Thread-safe lazy loading of model weights on first request
      - In-memory caching of loaded adapters (`AdapterCache`)
      - Hot-swapping model checkpoints with zero downtime
      - Metadata tracking & health monitoring across all disease modules
    """

    _instance: Optional[ModelRegistry] = None
    _singleton_lock = threading.Lock()

    def __init__(self, config: Optional[RegistryConfiguration] = None) -> None:
        self.config = config or RegistryConfiguration()
        self.logger = logging.getLogger("ModelRegistry")

        self.registry = AdapterRegistry()
        self.cache = AdapterCache()
        self.loader = AdapterLoader()
        self.factory = AdapterFactory()

        self._lock = threading.RLock()

        if self.config.auto_discover_adapters:
            self._auto_discover_built_in_adapters()

    @classmethod
    def get_instance(cls, config: Optional[RegistryConfiguration] = None) -> ModelRegistry:
        """Return global singleton instance of ModelRegistry."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(config=config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset global singleton instance (primarily for testing)."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.invalidate_cache()
                cls._instance = None

    def register_adapter(
        self,
        modality: str,
        adapter_cls: Type[IPredictorAdapter],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new predictor adapter class for a modality key.

        Args:
            modality: Lowercase modality key (e.g. 'skin').
            adapter_cls: Class implementing IPredictorAdapter.
            metadata: Optional metadata dict.
            overwrite: If True, replaces existing registration.
        """
        key = modality.lower()
        with self._lock:
            meta_obj = AdapterMetadata(modality=key, extra=metadata or {})
            route_cfg = self.config.get_modality_config(key)
            if route_cfg:
                meta_obj.model_path = route_cfg.get("checkpoint")
                meta_obj.label_map_path = route_cfg.get("label_map")

            self.registry.register(
                modality=key,
                adapter_cls=adapter_cls,
                metadata=meta_obj,
                overwrite=overwrite or self.config.allow_duplicate_registration,
            )
            self.logger.info(f"Registered adapter for modality '{key}' ({adapter_cls.__name__}).")

    def get_adapter(self, modality: str) -> IPredictorAdapter:
        """
        Retrieve a ready-to-predict adapter for the requested modality.
        Triggers thread-safe lazy loading if not already in cache.

        Args:
            modality: Modality key string.

        Returns:
            IPredictorAdapter instance.

        Raises:
            AdapterNotFoundException: If modality is unregistered.
            AdapterLoadException: If model weights fail to load.
        """
        key = modality.lower()
        with self._lock:
            # 1. Check cache first
            if self.config.enable_adapter_caching and self.cache.contains(key):
                cached = self.cache.get(key)
                if cached is not None:
                    return cached

            # 2. Get registered class
            adapter_cls = self.registry.get_class(key)
            route_cfg = self.config.get_modality_config(key)

            ckpt_path = route_cfg.get("checkpoint")
            label_map_path = route_cfg.get("label_map")

            # 3. Lazy load model weights
            self.logger.info(f"Lazy loading model weights for modality '{key}'...")
            try:
                adapter = self.loader.load(
                    adapter_cls=adapter_cls,
                    checkpoint_path=ckpt_path,
                    label_map_path=label_map_path,
                    config=route_cfg,
                )
            except Exception as e:
                meta = self.registry.get_metadata(key)
                meta.status = "ERRORED"
                meta.health = "UNHEALTHY"
                raise AdapterLoadException(modality=key, cause=e)

            # Update metadata
            meta = self.registry.get_metadata(key)
            meta.status = "LOADED"
            meta.health = "HEALTHY"
            try:
                meta.supported_classes = adapter.get_metadata().get("supported_classes", [])
            except Exception:
                pass

            # 4. Put in cache
            if self.config.enable_adapter_caching:
                self.cache.put(key, adapter)

            return adapter

    def is_registered(self, modality: str) -> bool:
        """Returns True if an adapter class is registered for modality."""
        return self.registry.is_registered(modality)

    def is_loaded(self, modality: str) -> bool:
        """Returns True if an adapter for modality is currently loaded in cache."""
        return self.cache.contains(modality)

    def list_registered_modalities(self) -> List[str]:
        """Return list of all registered modality keys."""
        return self.registry.list_modalities()

    def invalidate_cache(self, modality: Optional[str] = None) -> None:
        """
        Evict loaded adapter(s) from cache and release resources.
        """
        with self._lock:
            if modality is None:
                for k in self.registry.list_modalities():
                    try:
                        meta = self.registry.get_metadata(k)
                        meta.status = "UNLOADED"
                    except Exception:
                        pass
                self.cache.invalidate(None)
                self.logger.info("Invalidated all cached model adapters.")
            else:
                key = modality.lower()
                try:
                    meta = self.registry.get_metadata(key)
                    meta.status = "UNLOADED"
                except Exception:
                    pass
                self.cache.invalidate(key)
                self.logger.info(f"Invalidated cached adapter for modality '{key}'.")

    def reload_adapter(self, modality: str) -> IPredictorAdapter:
        """
        Unload and reload an adapter for a modality.
        """
        key = modality.lower()
        with self._lock:
            self.invalidate_cache(key)
            return self.get_adapter(key)

    def hot_swap(self, modality: str, new_checkpoint_path: str) -> IPredictorAdapter:
        """
        Atomic zero-downtime hot-swap of model weights for a modality.

        Loads new adapter weights in memory before replacing the cached pointer.
        """
        key = modality.lower()
        with self._lock:
            adapter_cls = self.registry.get_class(key)
            route_cfg = dict(self.config.get_modality_config(key))
            route_cfg["checkpoint"] = new_checkpoint_path

            self.logger.info(f"Hot-swapping model for modality '{key}' to checkpoint '{new_checkpoint_path}'...")
            new_adapter = self.loader.load(
                adapter_cls=adapter_cls,
                checkpoint_path=new_checkpoint_path,
                label_map_path=route_cfg.get("label_map"),
                config=route_cfg,
            )

            # Update cache pointer atomically
            self.cache.put(key, new_adapter)

            meta = self.registry.get_metadata(key)
            meta.model_path = new_checkpoint_path
            meta.status = "LOADED"
            meta.health = "HEALTHY"

            self.logger.info(f"Successfully hot-swapped modality '{key}'.")
            return new_adapter

    def get_adapter_metadata(self, modality: str) -> Dict[str, Any]:
        """Return metadata dictionary for a modality."""
        meta = self.registry.get_metadata(modality)
        return meta.to_dict()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check across all registered adapters.
        """
        with self._lock:
            modalities = self.list_registered_modalities()
            statuses = {}
            overall_status = "HEALTHY"

            for mod in modalities:
                is_lod = self.is_loaded(mod)
                meta = self.get_adapter_metadata(mod)

                statuses[mod] = {
                    "registered": True,
                    "loaded": is_lod,
                    "status": meta["status"],
                    "health": meta["health"],
                    "supported_classes": meta["supported_classes"],
                }

                if meta["health"] == "UNHEALTHY":
                    overall_status = "DEGRADED"

            return {
                "overall_status": overall_status,
                "registered_count": len(modalities),
                "loaded_count": self.cache.size(),
                "modalities": statuses,
            }

    def _auto_discover_built_in_adapters(self) -> None:
        """
        Auto-discover and register built-in Phase 2 disease adapters (Modules 7–13).
        Imports are isolated inside this helper method.
        """
        try:
            from medical_ai_engine.adapters.brain_mri_adapter import BrainMRIPredictorAdapter
            self.register_adapter("brain_mri", BrainMRIPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"BrainMRI adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.chest_xray_adapter import ChestXRayPredictorAdapter
            self.register_adapter("chest_xray", ChestXRayPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"ChestXRay adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.brain_ct_adapter import BrainCTPredictorAdapter
            self.register_adapter("brain_ct", BrainCTPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"BrainCT adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.histopathology_adapter import HistopathologyPredictorAdapter
            self.register_adapter("histopathology", HistopathologyPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"Histopathology adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.retina_adapter import RetinaPredictorAdapter
            self.register_adapter("retina", RetinaPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"Retina adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.skin_adapter import SkinPredictorAdapter
            self.register_adapter("skin", SkinPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"Skin adapter discovery skipped: {e}")

        try:
            from medical_ai_engine.adapters.ultrasound_adapter import UltrasoundPredictorAdapter
            self.register_adapter("ultrasound", UltrasoundPredictorAdapter)
        except Exception as e:
            self.logger.debug(f"Ultrasound adapter discovery skipped: {e}")
