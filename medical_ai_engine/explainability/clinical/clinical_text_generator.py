"""
clinical_text_generator.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Generates structured clinical-grade natural-language text.

Responsibility
--------------
Combines (disease_label, severity_label, region_descriptions, confidence_score,
modality) to produce a multi-sentence clinical impression paragraph suitable
for inclusion in a radiology-style AI report.
"""

from __future__ import annotations

import logging
import time
from typing import List

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import TextGenerationError

logger = logging.getLogger(__name__)

# ── Modality labels ───────────────────────────────────────────────────────────
_MODALITY_LABELS: dict[str, str] = {
    "brain_mri":       "Brain MRI",
    "chest_xray":      "Chest X-ray",
    "brain_ct":        "Brain CT",
    "histopathology":  "Histopathology",
    "retina":          "Retinal Fundus",
    "skin":            "Dermoscopy",
    "ultrasound":      "Ultrasound",
}

# ── Severity language modifiers ────────────────────────────────────────────────
_SEVERITY_MODIFIERS: dict[str, str] = {
    "critical":  "highly suspicious",
    "high":      "strongly suggestive",
    "moderate":  "suggestive",
    "low":       "mildly suggestive",
    "minimal":   "equivocal",
}

# ── Confidence qualifiers ─────────────────────────────────────────────────────
_CONFIDENCE_QUALIFIERS: list[tuple[float, str]] = [
    (0.90, "with high confidence"),
    (0.75, "with moderate-to-high confidence"),
    (0.60, "with moderate confidence"),
    (0.40, "with limited confidence"),
    (0.0,  "with low confidence"),
]

_CONFIDENCE_STMT = (
    "This analysis was performed by an AI system. Results should be reviewed "
    "by a qualified healthcare professional before clinical use."
)

_DISCLAIMER = (
    "⚠ FOR INVESTIGATIONAL USE ONLY. This AI-generated report is not approved "
    "for diagnostic decision-making without physician oversight."
)


def _confidence_qualifier(score: float) -> str:
    for threshold, label in _CONFIDENCE_QUALIFIERS:
        if score >= threshold:
            return label
    return "with low confidence"


class ClinicalTextGenerator:
    """
    Generates the clinical impression paragraph from the pipeline context.

    Output (standard verbosity) structure:
    1. Modality sentence: "Brain MRI analysis identified …"
    2. Severity sentence: "The AI finding is strongly suggestive of …"
    3. Region sentence:   "Prominent activation was observed in …"
    4. [Confidence statement if configured]
    """

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, ctx: ClinicalContext) -> str:
        """
        Generate clinical text from *ctx* and populate *ctx.clinical_text*.
        Returns the generated string.
        """
        t0 = time.perf_counter()
        try:
            text = self._compose(ctx)
            ctx.clinical_text = text
            ctx.record_stage_time(
                "text_generation",
                (time.perf_counter() - t0) * 1000,
            )
            return text
        except Exception as exc:
            msg = f"ClinicalTextGenerator failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise TextGenerationError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compose(self, ctx: ClinicalContext) -> str:
        modality_label = _MODALITY_LABELS.get(ctx.modality, ctx.modality.replace("_", " ").title())
        disease = ctx.disease_label.replace("_", " ").title()
        severity_mod = _SEVERITY_MODIFIERS.get(ctx.severity_label, "suggestive")
        conf_qual = _confidence_qualifier(ctx.confidence_score)
        region_descs: List[str] = ctx.region_descriptions or []

        # ── Sentence 1: Overview ──────────────────────────────────────────────
        s1 = (
            f"{modality_label} analysis identified imaging features "
            f"{severity_mod} of {disease} {conf_qual} "
            f"(confidence: {ctx.confidence_score:.1%})."
        )

        # ── Sentence 2: Severity ──────────────────────────────────────────────
        s2 = self._severity_sentence(ctx)

        # ── Sentence 3: Region summary ────────────────────────────────────────
        s3 = self._region_sentence(region_descs)

        paragraphs = [s1, s2, s3]

        if self._config.include_confidence_statement:
            paragraphs.append(_CONFIDENCE_STMT)

        if self._config.verbosity == "detailed":
            paragraphs.append(
                f"Explainability score (composite severity): {ctx.severity_score:.4f}. "
                f"Number of regions analysed: {len(ctx.classified_findings)}."
            )

        text = " ".join(p for p in paragraphs if p)
        if self._config.include_disclaimer:
            text = text + "\n\n" + _DISCLAIMER
        return text

    @staticmethod
    def _severity_sentence(ctx: ClinicalContext) -> str:
        sev = ctx.severity_label.capitalize() if ctx.severity_label else "Unknown"
        score = ctx.severity_score
        return (
            f"Severity assessment: {sev} (composite score: {score:.2f}). "
            f"Immediate clinical correlation is {'strongly ' if score >= 0.65 else ''}advised."
        )

    @staticmethod
    def _region_sentence(descs: List[str]) -> str:
        if not descs or descs == ["no significant focal activation detected"]:
            return "No focal activation patterns were identified in the Grad-CAM heatmap."
        if len(descs) == 1:
            return f"Activation map analysis revealed: {descs[0]}."
        primary = descs[0]
        secondary = "; ".join(descs[1:3])
        return (
            f"Activation map analysis revealed: {primary}. "
            f"Additional regions of interest: {secondary}."
        )


__all__ = ["ClinicalTextGenerator"]
