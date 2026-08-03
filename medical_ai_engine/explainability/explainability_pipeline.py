"""
Medical AI Engine — Explainability Pipeline

Module 15.1: Explainability Framework Architecture

Sequential pipeline orchestrating the 6 explainability execution stages:
  1. Context Validation & Setup
  2. Target Layer Resolution
  3. Activation Map Extraction (via CAM Method)
  4. Heatmap Generation
  5. Overlay Generation
  6. Clinical Explanation & Result Construction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from medical_ai_engine.explainability.cam_factory import CAMFactory
from medical_ai_engine.explainability.clinical_explanation_builder import ClinicalExplanationBuilder
from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration
from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.explainability_result import ExplainabilityResult
from medical_ai_engine.explainability.heatmap_generator import HeatmapGenerator
from medical_ai_engine.explainability.overlay_generator import OverlayGenerator
from medical_ai_engine.explainability.target_layer_manager import TargetLayerManager

logger = logging.getLogger("ExplainabilityPipeline")


class ExplainabilityPipeline:
    """
    Coordinates the 6-stage explainability workflow.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfiguration] = None,
        cam_factory: Optional[CAMFactory] = None,
        target_layer_manager: Optional[TargetLayerManager] = None,
        heatmap_generator: Optional[HeatmapGenerator] = None,
        overlay_generator: Optional[OverlayGenerator] = None,
        clinical_builder: Optional[ClinicalExplanationBuilder] = None,
    ) -> None:
        self.config = config or ExplainabilityConfiguration.default()
        self.factory = cam_factory or CAMFactory()
        self.layer_manager = target_layer_manager or TargetLayerManager(config=self.config)
        self.heatmap_generator = heatmap_generator or HeatmapGenerator(config=self.config)
        self.overlay_generator = overlay_generator or OverlayGenerator(config=self.config)
        self.clinical_builder = clinical_builder or ClinicalExplanationBuilder(config=self.config)

    def execute(self, ctx: ExplainabilityContext) -> ExplainabilityResult:
        """
        Run the 6-stage explainability pipeline.

        Args:
            ctx: Populated ExplainabilityContext.

        Returns:
            Immutable ExplainabilityResult object.
        """
        overall_start = time.perf_counter()
        ctx.execution_state = "RUNNING"
        ctx.log("Starting ExplainabilityPipeline execution...", stage="ExplainabilityPipeline")

        try:
            # Stage 1: Context Validation
            t0 = time.perf_counter()
            self._stage_1_validate(ctx)
            ctx.record_metric("stage1_validation_ms", (time.perf_counter() - t0) * 1000.0)

            # Stage 2: Target Layer Resolution
            t0 = time.perf_counter()
            self._stage_2_resolve_layer(ctx)
            ctx.record_metric("stage2_layer_resolution_ms", (time.perf_counter() - t0) * 1000.0)

            # Stage 3: Activation Map Extraction
            t0 = time.perf_counter()
            self._stage_3_extract_activation(ctx)
            ctx.record_metric("stage3_activation_ms", (time.perf_counter() - t0) * 1000.0)

            # Stage 4: Heatmap Generation
            t0 = time.perf_counter()
            self._stage_4_generate_heatmap(ctx)
            ctx.record_metric("stage4_heatmap_ms", (time.perf_counter() - t0) * 1000.0)

            # Stage 5: Overlay Generation
            t0 = time.perf_counter()
            self._stage_5_generate_overlay(ctx)
            ctx.record_metric("stage5_overlay_ms", (time.perf_counter() - t0) * 1000.0)

            # Stage 6: Clinical Explanation & Result Construction
            t0 = time.perf_counter()
            result = self._stage_6_build_result(ctx, overall_start)
            ctx.record_metric("stage6_result_build_ms", (time.perf_counter() - t0) * 1000.0)

            ctx.execution_state = "COMPLETED"
            ctx.log("ExplainabilityPipeline COMPLETED successfully.", stage="ExplainabilityPipeline")
            return result

        except Exception as exc:
            ctx.execution_state = "FAILED"
            ctx.add_error(stage="ExplainabilityPipeline", error=str(exc))
            logger.error("[ExplainabilityPipeline] Pipeline execution failed: %s", exc)
            return self._build_error_result(ctx, exc, overall_start)

    def _stage_1_validate(self, ctx: ExplainabilityContext) -> None:
        if not ctx.request_id:
            raise ValueError("Context missing request_id")
        ctx.log(f"Validated request '{ctx.request_id}' for modality '{ctx.modality}'", stage="Stage1Validation")

    def _stage_2_resolve_layer(self, ctx: ExplainabilityContext) -> None:
        layer_obj = self.layer_manager.resolve_target_layer(
            model=ctx.model_reference,
            modality=ctx.modality,
            explicit_layer_name=ctx.target_layer_name,
        )
        ctx.target_layer = layer_obj
        if not ctx.target_layer_name:
            ctx.target_layer_name = self.config.get_target_layer_for_modality(ctx.modality)
        ctx.log(f"Resolved target layer: '{ctx.target_layer_name}'", stage="Stage2LayerResolution")

    def _stage_3_extract_activation(self, ctx: ExplainabilityContext) -> None:
        cam_method = self.factory.create_cam_method(ctx.cam_method_key)
        activation_map = cam_method.generate_activation_map(
            model=ctx.model_reference,
            target_layer=ctx.target_layer,
            input_tensor=ctx.input_tensor,
            target_category=ctx.predicted_class_index,
        )
        ctx.activation_maps = activation_map
        ctx.log(f"Generated activation map using '{cam_method.method_name}'", stage="Stage3ActivationExtraction")

    def _stage_4_generate_heatmap(self, ctx: ExplainabilityContext) -> None:
        if ctx.activation_maps is None:
            raise ValueError("Activation map is None")
        heatmap_arr, pil_heatmap = self.heatmap_generator.generate_heatmap(
            activation_map=ctx.activation_maps,
            target_size=self.config.image_size,
            colormap_name=self.config.colormap,
        )
        ctx.heatmap_array = heatmap_arr
        ctx.log("Generated colormapped heatmap", stage="Stage4HeatmapGeneration")

    def _stage_5_generate_overlay(self, ctx: ExplainabilityContext) -> None:
        if ctx.image is not None and ctx.heatmap_array is not None:
            from PIL import Image
            pil_heatmap = Image.fromarray(ctx.heatmap_array, mode="RGB")
            overlay = self.overlay_generator.generate_overlay(
                original_image=ctx.image,
                heatmap_image=pil_heatmap,
                alpha=self.config.overlay_alpha,
            )
            ctx.overlay_image = overlay
            ctx.log("Generated alpha-blended overlay image", stage="Stage5OverlayGeneration")

    def _stage_6_build_result(self, ctx: ExplainabilityContext, overall_start: float) -> ExplainabilityResult:
        regions, summary = self.clinical_builder.build_explanation(
            activation_map=ctx.activation_maps if ctx.activation_maps is not None else np.zeros((224, 224)),
            modality=ctx.modality,
            predicted_disease=ctx.predicted_class,
            confidence=ctx.confidence,
        )
        total_time_ms = (time.perf_counter() - overall_start) * 1000.0

        return ExplainabilityResult(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            method=ctx.cam_method_key,
            modality=ctx.modality,
            predicted_class=ctx.predicted_class,
            confidence=ctx.confidence,
            target_layer=ctx.target_layer_name or "unknown",
            heatmap_path=ctx.heatmap_path,
            overlay_path=ctx.overlay_path,
            activation_map_path=ctx.activation_map_path,
            important_regions=regions,
            clinical_summary=summary,
            quality_score=100.0,
            generation_time_ms=total_time_ms,
            status="SUCCESS",
            performance_metrics=ctx.performance_metrics,
            warnings=ctx.warnings,
            errors=ctx.errors,
            extra=ctx.future_extensions,
        )

    def _build_error_result(
        self, ctx: ExplainabilityContext, exc: Exception, overall_start: float
    ) -> ExplainabilityResult:
        total_time_ms = (time.perf_counter() - overall_start) * 1000.0
        return ExplainabilityResult(
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            method=ctx.cam_method_key,
            modality=ctx.modality,
            predicted_class=ctx.predicted_class,
            confidence=ctx.confidence,
            target_layer=ctx.target_layer_name or "unknown",
            clinical_summary=f"Explainability generation failed: {exc}",
            quality_score=0.0,
            generation_time_ms=total_time_ms,
            status="FAILED",
            errors=ctx.errors or [{"stage": "Pipeline", "error": str(exc)}],
        )
