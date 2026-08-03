"""
Medical AI Engine — InputValidationStage (Stage 1)

Validates input image source (file path, PIL image, bytes), loads PIL Image into ctx.image,
and populates image metadata (dimensions, format, color_mode).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING
from PIL import Image

from medical_ai_engine.pipeline.pipeline_exceptions import StageValidationException
from medical_ai_engine.pipeline.stages.base_stage import AbstractPipelineStage

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class InputValidationStage(AbstractPipelineStage):
    """
    Stage 1: Input Validation and Image Ingestion.
    """

    def __init__(self) -> None:
        super().__init__("InputValidationStage")

    def validate(self, ctx: EngineContext) -> None:
        if ctx is None:
            raise StageValidationException("EngineContext cannot be None.", stage_name=self.name)
        if not ctx.request_id:
            raise StageValidationException("EngineContext missing request_id.", stage_name=self.name)

        has_image = ctx.image is not None
        has_source = ctx.image_source is not None
        has_bytes = ctx.image_bytes is not None

        if not (has_image or has_source or has_bytes):
            raise StageValidationException(
                "No image provided in request. Must specify image, image_source, or image_bytes.",
                stage_name=self.name,
            )

    def process(self, ctx: EngineContext) -> None:
        ctx.log("Validating and loading image input...", stage=self.name)

        pil_img: Image.Image | None = None

        if ctx.image is not None:
            if isinstance(ctx.image, Image.Image):
                pil_img = ctx.image
            else:
                raise TypeError(f"ctx.image is not a valid PIL Image: {type(ctx.image)}")
        elif ctx.image_source is not None:
            path = Path(ctx.image_source)
            if not path.is_file():
                raise FileNotFoundError(f"Image source file not found: {path}")
            pil_img = Image.open(path)
        elif ctx.image_bytes is not None:
            pil_img = Image.open(io.BytesIO(ctx.image_bytes))

        if pil_img is None:
            raise StageValidationException("Failed to decode input image.", stage_name=self.name)

        # Convert palette/RGBA modes to RGB if needed for universal pipeline consumption
        if pil_img.mode in ("RGBA", "P", "LA"):
            pil_img = pil_img.convert("RGB")

        ctx.image = pil_img
        ctx.metadata["format"] = getattr(pil_img, "format", "PNG") or "PNG"
        ctx.metadata["size"] = pil_img.size  # (width, height)
        ctx.metadata["color_mode"] = pil_img.mode

        ctx.set_timestamp("validation_complete")
        ctx.log(
            f"Image validated successfully: size={pil_img.size}, mode={pil_img.mode}",
            stage=self.name,
        )

    def rollback(self, ctx: EngineContext) -> None:
        ctx.log("Rolling back InputValidationStage...", stage=self.name)
        ctx.metadata.pop("format", None)
        ctx.metadata.pop("size", None)
        ctx.metadata.pop("color_mode", None)
