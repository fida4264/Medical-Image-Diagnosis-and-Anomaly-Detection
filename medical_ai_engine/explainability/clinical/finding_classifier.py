"""
finding_classifier.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Classifies anatomical / pathological findings per modality.

Responsibility
--------------
Maps (modality, disease_label, activation_regions) → a list of structured
finding dictionaries with anatomy, pathology type, and clinical significance.
Does NOT produce natural-language sentences (that is ClinicalTextGenerator's job).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import FindingClassificationError

logger = logging.getLogger(__name__)

# ── Modality → anatomy vocabulary ────────────────────────────────────────────
_ANATOMY_MAP: Dict[str, Dict[str, str]] = {
    "brain_mri": {
        "default": "brain parenchyma",
        "frontal": "frontal lobe",
        "temporal": "temporal lobe",
        "parietal": "parietal lobe",
        "occipital": "occipital lobe",
        "cerebellum": "cerebellum",
        "brainstem": "brainstem",
        "basal_ganglia": "basal ganglia",
        "white_matter": "white matter",
    },
    "chest_xray": {
        "default": "lung field",
        "upper": "upper lobe",
        "middle": "middle lobe",
        "lower": "lower lobe",
        "hilar": "hilar region",
        "pleural": "pleural space",
        "cardiac": "cardiac silhouette",
        "mediastinum": "mediastinum",
    },
    "brain_ct": {
        "default": "intracranial compartment",
        "cortex": "cortical region",
        "ventricle": "ventricular system",
        "subdural": "subdural space",
        "subarachnoid": "subarachnoid space",
        "epidural": "epidural space",
        "intraparenchymal": "brain parenchyma",
    },
    "histopathology": {
        "default": "tissue section",
        "gland": "glandular tissue",
        "stroma": "stromal compartment",
        "nuclear": "nuclear region",
        "cytoplasm": "cytoplasmic region",
    },
    "retina": {
        "default": "retinal field",
        "macula": "macular region",
        "optic_disc": "optic disc",
        "peripheral": "peripheral retina",
        "vessel": "retinal vasculature",
        "fovea": "foveal region",
    },
    "skin": {
        "default": "skin lesion",
        "center": "lesion center",
        "border": "lesion border",
        "peripheral": "peripheral skin",
    },
    "ultrasound": {
        "default": "echogenic region",
        "hypoechoic": "hypoechoic focus",
        "hyperechoic": "hyperechoic focus",
        "cystic": "cystic region",
        "solid": "solid mass",
    },
}

# ── Pathology type vocabulary ─────────────────────────────────────────────────
_PATHOLOGY_MAP: Dict[str, str] = {
    # Brain MRI
    "tumor": "neoplastic lesion",
    "glioma": "glial tumor",
    "meningioma": "meningeal neoplasm",
    "pituitary": "pituitary mass",
    "alzheimer": "neurodegenerative change",
    # Chest X-Ray
    "pneumonia": "consolidative opacity",
    "covid": "bilateral ground-glass opacification",
    "tuberculosis": "cavitary infiltrate",
    "pleural_effusion": "pleural fluid collection",
    "cardiomegaly": "cardiomegaly",
    "lung_cancer": "pulmonary mass",
    # Brain CT
    "hemorrhage": "hyperdense collection",
    "epidural": "lenticular hyperdensity",
    "subdural": "crescent-shaped hyperdensity",
    "stroke": "hypodense ischemic territory",
    # Histopathology
    "cancer": "malignant cellular proliferation",
    "adenocarcinoma": "glandular malignancy",
    "squamous": "squamous cell carcinoma",
    "normal": "no significant histologic abnormality",
    # Retina
    "diabetic_retinopathy": "diabetic microangiopathy",
    "glaucoma": "optic neuropathy",
    "cataract": "lens opacity",
    "macular_degeneration": "macular degeneration",
    # Skin
    "melanoma": "malignant melanocytic lesion",
    "basal_cell": "basal cell carcinoma",
    "nevus": "melanocytic nevus",
    # Ultrasound
    "benign": "benign focal lesion",
    "malignant": "malignant focal lesion",
    "thyroid_nodule": "thyroid nodule",
}


class FindingClassifier:
    """
    Classifies anatomical findings from heatmap regions and disease predictions.

    Each finding is returned as a dict with:
        - anatomy:       anatomical location label
        - pathology:     pathology type description
        - peak_score:    peak activation intensity [0, 1]
        - area_fraction: fraction of image covered [0, 1]
        - region_index:  index into the important_regions list
        - significance:  'primary' | 'secondary' | 'incidental'
    """

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, ctx: ClinicalContext) -> List[Dict[str, Any]]:
        """
        Classify findings from *ctx* and populate *ctx.classified_findings*.
        Returns the list of classified finding dicts.
        """
        t0 = time.perf_counter()
        try:
            findings = self._run_classification(ctx)
            ctx.classified_findings = findings
            ctx.record_stage_time(
                "finding_classification",
                (time.perf_counter() - t0) * 1000,
            )
            return findings
        except Exception as exc:
            msg = f"FindingClassifier failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise FindingClassificationError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_classification(self, ctx: ClinicalContext) -> List[Dict[str, Any]]:
        modality = ctx.modality or "brain_mri"
        disease = ctx.disease_label.lower().replace(" ", "_").replace("-", "_")
        anatomy_vocab = _ANATOMY_MAP.get(modality, _ANATOMY_MAP["brain_mri"])
        pathology_type = self._resolve_pathology(disease)

        findings: List[Dict[str, Any]] = []

        regions = ctx.important_regions or []
        if not regions:
            # Synthesize a single primary finding from the disease prediction alone
            findings.append(self._make_finding(
                anatomy=anatomy_vocab["default"],
                pathology=pathology_type,
                peak_score=ctx.confidence_score,
                area_fraction=0.0,
                region_index=-1,
                significance="primary",
            ))
            return findings

        for idx, region in enumerate(regions[: self._config.max_findings_per_report]):
            peak = float(region.get("peak_activation", ctx.confidence_score))
            area = float(region.get("area_ratio", 0.0))
            anatomy = self._resolve_anatomy(anatomy_vocab, region)
            significance = "primary" if idx == 0 else ("secondary" if idx <= 2 else "incidental")
            findings.append(self._make_finding(
                anatomy=anatomy,
                pathology=pathology_type,
                peak_score=peak,
                area_fraction=area,
                region_index=idx,
                significance=significance,
            ))

        return findings

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_finding(
        anatomy: str,
        pathology: str,
        peak_score: float,
        area_fraction: float,
        region_index: int,
        significance: str,
    ) -> Dict[str, Any]:
        return {
            "anatomy": anatomy,
            "pathology": pathology,
            "peak_score": round(peak_score, 4),
            "area_fraction": round(area_fraction, 4),
            "region_index": region_index,
            "significance": significance,
        }

    @staticmethod
    def _resolve_pathology(disease: str) -> str:
        for key, label in _PATHOLOGY_MAP.items():
            if key in disease:
                return label
        # Fallback: title-case the label as-is
        return disease.replace("_", " ").title()

    @staticmethod
    def _resolve_anatomy(vocab: Dict[str, str], region: Dict[str, Any]) -> str:
        # Try to match centroid position to a rough anatomical zone
        cx = region.get("centroid_x", 0.5)
        cy = region.get("centroid_y", 0.5)
        # Rough spatial heuristics
        if cy < 0.33:
            return vocab.get("upper", vocab.get("frontal", vocab["default"]))
        if cy > 0.66:
            return vocab.get("lower", vocab.get("occipital", vocab["default"]))
        return vocab["default"]


__all__ = ["FindingClassifier"]
