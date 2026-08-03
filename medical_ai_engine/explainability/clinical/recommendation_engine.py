"""
recommendation_engine.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Generates modality & severity-aware clinical action recommendations.

Responsibility
--------------
Produces an ordered list of actionable clinical recommendations based on
(modality, disease_label, severity_label, confidence_score).
Recommendations are evidence-informed template strings; they are NOT
diagnostic statements and must be validated by a clinician.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import RecommendationEngineError

logger = logging.getLogger(__name__)

# ── Recommendation templates ──────────────────────────────────────────────────
# Key: (modality, severity_level) → list[str]
# severity_level: 'critical' | 'high' | 'moderate' | 'low' | 'minimal'
_RECOMMENDATIONS: Dict[Tuple[str, str], List[str]] = {
    # ── Brain MRI ─────────────────────────────────────────────────────────────
    ("brain_mri", "critical"): [
        "Urgent neurosurgical or neuro-oncology consultation recommended.",
        "Consider gadolinium-enhanced MRI for lesion characterisation.",
        "Functional MRI (fMRI) and MR spectroscopy may be indicated.",
        "Cross-sectional neurology review within 24 hours advised.",
    ],
    ("brain_mri", "high"): [
        "Neurology referral within 72 hours is advised.",
        "Consider gadolinium-enhanced MRI for lesion characterisation.",
        "Follow-up MRI in 4–6 weeks to assess lesion stability.",
    ],
    ("brain_mri", "moderate"): [
        "Neurology outpatient review within 4 weeks recommended.",
        "Follow-up MRI in 3 months.",
    ],
    ("brain_mri", "low"): [
        "Routine follow-up MRI in 6 months.",
        "Clinical correlation with neurological examination.",
    ],
    ("brain_mri", "minimal"): [
        "No immediate action required. Routine clinical follow-up.",
    ],
    # ── Chest X-Ray ──────────────────────────────────────────────────────────
    ("chest_xray", "critical"): [
        "Immediate chest CT with contrast recommended.",
        "Urgent respiratory/pulmonary medicine referral.",
        "Consider bronchoscopy and sputum culture if infective aetiology suspected.",
    ],
    ("chest_xray", "high"): [
        "Chest CT within 48 hours recommended.",
        "Pulmonology referral within 72 hours.",
        "Serial chest X-rays at 4-week intervals.",
    ],
    ("chest_xray", "moderate"): [
        "Follow-up chest X-ray in 6–8 weeks.",
        "Clinical correlation with history and physical examination.",
    ],
    ("chest_xray", "low"): [
        "Routine follow-up chest X-ray in 3 months.",
    ],
    ("chest_xray", "minimal"): [
        "No significant imaging abnormality detected. Routine annual review.",
    ],
    # ── Brain CT ─────────────────────────────────────────────────────────────
    ("brain_ct", "critical"): [
        "Emergency neurosurgical consultation required.",
        "Consider immediate CT angiography (CTA) for vascular aetiology.",
        "Intensive care unit (ICU) monitoring indicated.",
    ],
    ("brain_ct", "high"): [
        "Urgent neurology review within 12 hours.",
        "MRI brain with DWI sequences recommended for characterisation.",
    ],
    ("brain_ct", "moderate"): [
        "Neurology outpatient follow-up within 2 weeks.",
        "Repeat CT in 48–72 hours to assess evolution.",
    ],
    ("brain_ct", "low"): [
        "Routine neurology follow-up.",
        "Repeat CT in 4 weeks if clinically indicated.",
    ],
    ("brain_ct", "minimal"): [
        "No significant intracranial abnormality detected.",
    ],
    # ── Histopathology ───────────────────────────────────────────────────────
    ("histopathology", "critical"): [
        "Multidisciplinary tumour board review urgently recommended.",
        "Immunohistochemistry panel to confirm malignant lineage.",
        "Molecular profiling (NGS) for therapeutic target identification.",
        "Oncology staging workup to be initiated.",
    ],
    ("histopathology", "high"): [
        "Pathologist senior review and sign-out recommended.",
        "Additional immunohistochemistry markers as clinically indicated.",
        "Oncology referral within 1 week.",
    ],
    ("histopathology", "moderate"): [
        "Pathology correlation with clinical and radiological findings.",
        "Follow-up biopsy in 3–6 months if indicated.",
    ],
    ("histopathology", "low"): [
        "Routine pathology follow-up.",
    ],
    ("histopathology", "minimal"): [
        "No significant histologic abnormality. Routine surveillance.",
    ],
    # ── Retina ───────────────────────────────────────────────────────────────
    ("retina", "critical"): [
        "Urgent ophthalmology referral within 24 hours.",
        "Consider fluorescein angiography and OCT imaging.",
        "Systemic disease management (e.g., diabetes, hypertension) optimisation.",
    ],
    ("retina", "high"): [
        "Ophthalmology review within 1 week.",
        "Optical coherence tomography (OCT) recommended.",
        "Blood glucose and HbA1c measurement if diabetic aetiology suspected.",
    ],
    ("retina", "moderate"): [
        "Ophthalmology follow-up within 1 month.",
        "Fundus photography for documentation and monitoring.",
    ],
    ("retina", "low"): [
        "Routine ophthalmology review in 6 months.",
    ],
    ("retina", "minimal"): [
        "No significant retinal abnormality. Annual eye examination.",
    ],
    # ── Skin ─────────────────────────────────────────────────────────────────
    ("skin", "critical"): [
        "Urgent dermatology / plastic surgery referral.",
        "Wide local excision with sentinel lymph node biopsy may be indicated.",
        "Full-body skin examination and dermoscopy.",
        "Consider systemic staging (PET-CT) if melanoma suspected.",
    ],
    ("skin", "high"): [
        "Dermatology referral within 2 weeks.",
        "Excision biopsy with histopathology recommended.",
    ],
    ("skin", "moderate"): [
        "Dermatology outpatient review within 1 month.",
        "Photographic documentation for surveillance.",
    ],
    ("skin", "low"): [
        "Routine dermatology review.",
        "Self-monitoring for changes in size, shape, or colour.",
    ],
    ("skin", "minimal"): [
        "No clinically significant lesion characteristics identified.",
    ],
    # ── Ultrasound ───────────────────────────────────────────────────────────
    ("ultrasound", "critical"): [
        "Urgent interventional radiology or surgical referral.",
        "Consider CT or MRI for further lesion characterisation.",
        "Image-guided biopsy may be required for tissue diagnosis.",
    ],
    ("ultrasound", "high"): [
        "Radiology review within 48 hours.",
        "Follow-up ultrasound in 4–6 weeks.",
        "Cross-sectional imaging (CT/MRI) for complete characterisation.",
    ],
    ("ultrasound", "moderate"): [
        "Follow-up ultrasound in 3 months.",
        "Clinical correlation recommended.",
    ],
    ("ultrasound", "low"): [
        "Routine ultrasound surveillance in 6 months.",
    ],
    ("ultrasound", "minimal"): [
        "No significant sonographic abnormality detected.",
    ],
}

_DEFAULT_RECOMMENDATIONS: List[str] = [
    "Clinical correlation with history and physical examination is recommended.",
    "Results should be reviewed by a qualified healthcare professional.",
]


class RecommendationEngine:
    """
    Generates modality- and severity-stratified clinical action recommendations.
    """

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self, ctx: ClinicalContext) -> List[str]:
        """
        Generate recommendations from *ctx* and populate *ctx.recommendations*.
        Returns the list of recommendation strings.
        """
        if not self._config.include_recommendations:
            ctx.recommendations = []
            return []

        t0 = time.perf_counter()
        try:
            recs = self._build(ctx)
            ctx.recommendations = recs
            ctx.record_stage_time(
                "recommendation_generation",
                (time.perf_counter() - t0) * 1000,
            )
            return recs
        except Exception as exc:
            msg = f"RecommendationEngine failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise RecommendationEngineError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(self, ctx: ClinicalContext) -> List[str]:
        key = (ctx.modality, ctx.severity_label)
        recs = list(_RECOMMENDATIONS.get(key, _DEFAULT_RECOMMENDATIONS))

        # Always append a universal disclaimer as the last recommendation
        universal = (
            "All findings generated by this AI system require validation by a "
            "licensed healthcare professional prior to clinical decision-making."
        )
        if universal not in recs:
            recs.append(universal)

        return recs


__all__ = ["RecommendationEngine"]
