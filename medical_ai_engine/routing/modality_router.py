"""
Medical AI Engine — Modality Router Implementation

Core orchestrator of Module 14.2.3.
Implements IModalityRouter interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from PIL import Image

from modality_classifier.inference.predictor import ModalityPredictor
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.interfaces.router_interface import IModalityRouter
from medical_ai_engine.registry.model_registry import ModelRegistry
from medical_ai_engine.routing.routing_configuration import RoutingConfiguration
from medical_ai_engine.routing.routing_decision import (
    RoutingDecision,
    RoutingOutcome,
    make_route_to_model,
    make_routing_failed,
)
from medical_ai_engine.routing.routing_exceptions import (
    ModalityClassificationException,
    RoutingException,
)
from medical_ai_engine.routing.routing_policy import RoutingPolicy


class ModalityRouter(IModalityRouter):
    """
    Intelligent Modality Router for Medical AI Platform.

    Responsibilities:
      - Receives EngineContext
      - Checks manual_modality_override
      - Invokes Module 6 ModalityPredictor on input image
      - Evaluates RoutingPolicy (modality support, confidence thresholds, adapter existence)
      - Queries ModelRegistry (without direct disease model imports)
      - Populates ctx.modality_prediction
      - Returns RoutingDecision
    """

    def __init__(
        self,
        config: Optional[RoutingConfiguration] = None,
        model_registry: Optional[ModelRegistry] = None,
        predictor: Optional[ModalityPredictor] = None,
        policy: Optional[RoutingPolicy] = None,
    ) -> None:
        self.config = config or RoutingConfiguration.load()
        self.logger = logging.getLogger(self.config.router_name)

        self.registry = model_registry or ModelRegistry.get_instance()
        self.policy = policy or RoutingPolicy(config=self.config)

        self._predictor = predictor

    @property
    def predictor(self) -> ModalityPredictor:
        """Lazy-load ModalityPredictor instance on demand."""
        if self._predictor is None:
            try:
                self._predictor = ModalityPredictor(
                    checkpoint_path=self.config.modality_classifier_checkpoint,
                    label_map_path=self.config.modality_classifier_label_map,
                    device=self.config.modality_classifier_device,
                    target_size=self.config.modality_classifier_target_size,
                )
            except Exception as e:
                self.logger.warning(
                    f"ModalityPredictor failed to load weights ({e}). "
                    "Unweighted model initialized."
                )
                self._predictor = ModalityPredictor(
                    checkpoint_path="nonexistent.pt",
                    label_map_path=self.config.modality_classifier_label_map,
                    target_size=self.config.modality_classifier_target_size,
                )
        return self._predictor

    def route(self, ctx: EngineContext) -> RoutingDecision:
        """
        Execute routing for the given EngineContext.
        """
        t0 = time.perf_counter()
        req_id = ctx.request_id

        ctx.log("Starting modality routing...", stage="ModalityRouter")

        try:
            # 1. Check for manual modality override
            if self.config.allow_manual_override and ctx.manual_modality_override:
                override_mod = ctx.manual_modality_override.lower()
                ctx.log(f"Manual modality override detected: '{override_mod}'", stage="ModalityRouter")

                routing_time_ms = (time.perf_counter() - t0) * 1000.0
                is_registered = self.registry.is_registered(override_mod)

                # Apply policy on override
                rejection = self.policy.apply(
                    modality=override_mod,
                    confidence=1.0,
                    adapter_registered=is_registered,
                    request_id=req_id,
                    routing_time_ms=routing_time_ms,
                )

                if rejection:
                    decision = rejection
                else:
                    decision = make_route_to_model(
                        request_id=req_id,
                        modality=override_mod,
                        confidence=1.0,
                        adapter_key=override_mod,
                        threshold=self.config.get_threshold(override_mod),
                        routing_mode="manual_override",
                        routing_time_ms=routing_time_ms,
                    )

                self._populate_context(ctx, decision)
                return decision

            # 2. Extract PIL image from ctx
            image_input = self._extract_image(ctx)

            # 3. Invoke Module 6 Modality Classifier
            try:
                mod_result = self.predictor.predict(
                    image_input=image_input,
                    top_k=self.config.top_k,
                )
            except Exception as cls_err:
                raise ModalityClassificationException(
                    f"Modality classifier prediction failed: {cls_err}",
                    cause=cls_err,
                )

            detected_mod = mod_result.predicted_modality.lower()
            confidence = float(mod_result.confidence)
            top_k_preds = mod_result.top_k_predictions

            ctx.log(
                f"Modality detected: '{detected_mod}' (confidence: {confidence:.4f})",
                stage="ModalityRouter",
            )

            routing_time_ms = (time.perf_counter() - t0) * 1000.0
            is_registered = self.registry.is_registered(detected_mod)

            # 4. Apply Routing Policy (support check, confidence threshold, adapter registration)
            rejection = self.policy.apply(
                modality=detected_mod,
                confidence=confidence,
                adapter_registered=is_registered,
                request_id=req_id,
                routing_time_ms=routing_time_ms,
                top_k=top_k_preds,
            )

            if rejection is not None:
                decision = rejection
            else:
                threshold = self.config.get_threshold(detected_mod)
                warnings = []
                if (
                    self.config.warn_on_low_confidence
                    and confidence < self.config.confidence_warn_threshold
                ):
                    warn_msg = (
                        f"Modality '{detected_mod}' confidence {confidence:.3f} "
                        f"is below warning threshold {self.config.confidence_warn_threshold:.2f}."
                    )
                    warnings.append(warn_msg)
                    ctx.warn(warn_msg, stage="ModalityRouter")

                decision = make_route_to_model(
                    request_id=req_id,
                    modality=detected_mod,
                    confidence=confidence,
                    adapter_key=detected_mod,
                    threshold=threshold,
                    routing_mode="automatic",
                    routing_time_ms=routing_time_ms,
                    top_k=top_k_preds,
                    warnings=warnings,
                )

            self._populate_context(ctx, decision)
            return decision

        except Exception as e:
            routing_time_ms = (time.perf_counter() - t0) * 1000.0
            self.logger.error(f"Routing failed for request {req_id}: {e}")
            decision = make_routing_failed(
                request_id=req_id,
                modality=(ctx.manual_modality_override or "unknown").lower(),
                confidence=0.0,
                error_message=str(e),
                routing_time_ms=routing_time_ms,
            )
            self._populate_context(ctx, decision)
            return decision

    def _extract_image(self, ctx: EngineContext) -> Any:
        """Extract PIL image or image source path/bytes from EngineContext."""
        if ctx.image is not None:
            return ctx.image
        if ctx.image_source is not None:
            return ctx.image_source
        if ctx.image_bytes is not None:
            import io
            return Image.open(io.BytesIO(ctx.image_bytes))

        # Dummy fallback image for testing context with no image attached
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def _populate_context(self, ctx: EngineContext, decision: RoutingDecision) -> None:
        """Update EngineContext with routing output."""
        ctx.modality_prediction = {
            "detected_modality": decision.detected_modality,
            "modality_confidence": decision.confidence,
            "routing_outcome": decision.outcome.value,
            "adapter_key": decision.adapter_key,
            "routing_mode": decision.routing_mode,
            "routing_time_ms": decision.routing_time_ms,
            "reason": decision.reason,
            "recommended_action": decision.recommended_action,
            "threshold_applied": decision.threshold_applied,
            "top_k_modalities": decision.top_k_modalities,
        }
        ctx.set_timestamp("modality_predicted")
        ctx.record_metric("routing_ms", decision.routing_time_ms)
        ctx.log(
            f"Routing decision: {decision.outcome.value} → modality '{decision.detected_modality}'",
            stage="ModalityRouter",
        )

        if decision.warnings:
            for w in decision.warnings:
                ctx.warn(w, stage="ModalityRouter")

        if decision.outcome.is_failure:
            ctx.add_error(stage="ModalityRouter", error=decision.reason)

    def health_check(self) -> Dict[str, Any]:
        """Return Health Check dictionary."""
        reg_health = self.registry.health_check()
        return {
            "status": "HEALTHY" if reg_health["overall_status"] == "HEALTHY" else "DEGRADED",
            "router_name": self.config.router_name,
            "router_version": self.config.router_version,
            "modality_classifier_loaded": self._predictor is not None,
            "registry_available": True,
            "supported_modalities": self.config.supported_modalities,
            "confidence_thresholds": self.config.confidence_thresholds,
            "registry_health": reg_health,
        }
