"""
test_clinical_explanation.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Unit tests: 20 tests across all 6 pipeline stages.
"""

from __future__ import annotations

import pytest

from medical_ai_engine.explainability.clinical import (
    ClinicalConfiguration,
    ClinicalContext,
    ClinicalExplanationEngine,
    ClinicalReportBuilder,
    ClinicalTextGenerator,
    FindingClassifier,
    RecommendationEngine,
    RegionDescriptor,
    SeverityAssessor,
)
from medical_ai_engine.explainability.clinical.clinical_exceptions import (
    ConfigurationError,
    SeverityAssessmentError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> ClinicalConfiguration:
    return ClinicalConfiguration()


@pytest.fixture
def brain_mri_ctx() -> ClinicalContext:
    return ClinicalContext(
        modality="brain_mri",
        disease_label="glioma",
        confidence_score=0.92,
        image_width=224,
        image_height=224,
        important_regions=[
            {"peak_activation": 0.95, "area_ratio": 0.35, "centroid_x": 0.3, "centroid_y": 0.25},
            {"peak_activation": 0.72, "area_ratio": 0.12, "centroid_x": 0.7, "centroid_y": 0.60},
        ],
        visualization_metrics={"coverage_ratio": 0.28, "sharpness": 0.85},
    )


@pytest.fixture
def chest_ctx() -> ClinicalContext:
    return ClinicalContext(
        modality="chest_xray",
        disease_label="pneumonia",
        confidence_score=0.78,
        important_regions=[
            {"peak_activation": 0.80, "area_ratio": 0.22, "centroid_x": 0.5, "centroid_y": 0.7},
        ],
        visualization_metrics={"coverage_ratio": 0.22},
    )


@pytest.fixture
def engine(default_config) -> ClinicalExplanationEngine:
    return ClinicalExplanationEngine(config=default_config)


# ── T01 — ClinicalConfiguration defaults ─────────────────────────────────────

class TestClinicalConfiguration:

    def test_default_creation(self):
        cfg = ClinicalConfiguration()
        assert cfg.language == "en"
        assert cfg.verbosity == "standard"
        assert cfg.max_findings_per_report == 5

    def test_invalid_verbosity_raises(self):
        with pytest.raises(ConfigurationError):
            ClinicalConfiguration(verbosity="verbose")

    def test_invalid_language_raises(self):
        with pytest.raises(ConfigurationError):
            ClinicalConfiguration(language="fr")

    def test_severity_classification(self):
        cfg = ClinicalConfiguration()
        assert cfg.classify_severity(0.90) == "critical"
        assert cfg.classify_severity(0.70) == "high"
        assert cfg.classify_severity(0.50) == "moderate"
        assert cfg.classify_severity(0.25) == "low"
        assert cfg.classify_severity(0.05) == "minimal"

    def test_region_size_classification(self):
        cfg = ClinicalConfiguration()
        assert cfg.classify_region_size(0.45) == "extensive"
        assert cfg.classify_region_size(0.28) == "large"
        assert cfg.classify_region_size(0.14) == "moderate"
        assert cfg.classify_region_size(0.07) == "small"
        assert cfg.classify_region_size(0.01) == "focal"

    def test_to_dict_keys(self):
        cfg = ClinicalConfiguration()
        d = cfg.to_dict()
        assert "language" in d
        assert "verbosity" in d
        assert "max_findings_per_report" in d


# ── T02 — FindingClassifier ───────────────────────────────────────────────────

class TestFindingClassifier:

    def test_classify_with_regions(self, default_config, brain_mri_ctx):
        classifier = FindingClassifier(default_config)
        findings = classifier.classify(brain_mri_ctx)
        assert len(findings) >= 1
        assert "anatomy" in findings[0]
        assert "pathology" in findings[0]
        assert "peak_score" in findings[0]
        assert "significance" in findings[0]

    def test_primary_finding_significance(self, default_config, brain_mri_ctx):
        classifier = FindingClassifier(default_config)
        findings = classifier.classify(brain_mri_ctx)
        assert findings[0]["significance"] == "primary"

    def test_classify_without_regions(self, default_config):
        ctx = ClinicalContext(
            modality="chest_xray",
            disease_label="pneumonia",
            confidence_score=0.65,
        )
        classifier = FindingClassifier(default_config)
        findings = classifier.classify(ctx)
        assert len(findings) == 1
        assert findings[0]["region_index"] == -1

    def test_context_populated(self, default_config, brain_mri_ctx):
        classifier = FindingClassifier(default_config)
        classifier.classify(brain_mri_ctx)
        assert len(brain_mri_ctx.classified_findings) >= 1
        assert "finding_classification" in brain_mri_ctx.stage_timings_ms


# ── T03 — SeverityAssessor ────────────────────────────────────────────────────

class TestSeverityAssessor:

    def test_severity_label_set(self, default_config, brain_mri_ctx):
        FindingClassifier(default_config).classify(brain_mri_ctx)
        assessor = SeverityAssessor(default_config)
        label = assessor.assess(brain_mri_ctx)
        assert label in {"critical", "high", "moderate", "low", "minimal"}

    def test_high_confidence_gives_high_severity(self, default_config):
        ctx = ClinicalContext(
            modality="brain_mri",
            disease_label="tumor",
            confidence_score=0.95,
            visualization_metrics={"coverage_ratio": 0.45},
        )
        FindingClassifier(default_config).classify(ctx)
        SeverityAssessor(default_config).assess(ctx)
        assert ctx.severity_label in {"critical", "high"}

    def test_low_confidence_gives_low_severity(self, default_config):
        ctx = ClinicalContext(
            modality="skin",
            disease_label="nevus",
            confidence_score=0.15,
            visualization_metrics={"coverage_ratio": 0.03},
        )
        FindingClassifier(default_config).classify(ctx)
        SeverityAssessor(default_config).assess(ctx)
        assert ctx.severity_label in {"minimal", "low"}


# ── T04 — RegionDescriptor ────────────────────────────────────────────────────

class TestRegionDescriptor:

    def test_descriptions_generated(self, default_config, brain_mri_ctx):
        FindingClassifier(default_config).classify(brain_mri_ctx)
        SeverityAssessor(default_config).assess(brain_mri_ctx)
        describer = RegionDescriptor(default_config)
        descs = describer.describe(brain_mri_ctx)
        assert len(descs) >= 1
        assert isinstance(descs[0], str)
        assert len(descs[0]) > 10

    def test_brief_verbosity(self, default_config, brain_mri_ctx):
        cfg = ClinicalConfiguration(verbosity="brief")
        FindingClassifier(cfg).classify(brain_mri_ctx)
        SeverityAssessor(cfg).assess(brain_mri_ctx)
        descs = RegionDescriptor(cfg).describe(brain_mri_ctx)
        # brief format should not contain parenthetical area size
        assert all("(" not in d for d in descs), f"Expected brief format but got: {descs}"

    def test_no_regions_fallback(self, default_config):
        ctx = ClinicalContext(modality="retina", disease_label="glaucoma", confidence_score=0.55)
        FindingClassifier(default_config).classify(ctx)
        SeverityAssessor(default_config).assess(ctx)
        descs = RegionDescriptor(default_config).describe(ctx)
        assert len(descs) == 1


# ── T05 — ClinicalTextGenerator ──────────────────────────────────────────────

class TestClinicalTextGenerator:

    def _run_pipeline_to_text(self, ctx, cfg=None):
        cfg = cfg or ClinicalConfiguration()
        FindingClassifier(cfg).classify(ctx)
        SeverityAssessor(cfg).assess(ctx)
        RegionDescriptor(cfg).describe(ctx)
        ClinicalTextGenerator(cfg).generate(ctx)

    def test_clinical_text_nonempty(self, default_config, brain_mri_ctx):
        self._run_pipeline_to_text(brain_mri_ctx)
        assert len(brain_mri_ctx.clinical_text) > 50

    def test_text_contains_disease(self, default_config, brain_mri_ctx):
        self._run_pipeline_to_text(brain_mri_ctx)
        assert "Glioma" in brain_mri_ctx.clinical_text or "glioma" in brain_mri_ctx.clinical_text

    def test_disclaimer_appended(self, default_config, chest_ctx):
        self._run_pipeline_to_text(chest_ctx)
        assert "FOR INVESTIGATIONAL USE ONLY" in chest_ctx.clinical_text


# ── T06 — RecommendationEngine ────────────────────────────────────────────────

class TestRecommendationEngine:

    def _run_to_recommendations(self, ctx, cfg=None):
        cfg = cfg or ClinicalConfiguration()
        FindingClassifier(cfg).classify(ctx)
        SeverityAssessor(cfg).assess(ctx)
        RegionDescriptor(cfg).describe(ctx)
        ClinicalTextGenerator(cfg).generate(ctx)
        return RecommendationEngine(cfg).recommend(ctx)

    def test_recommendations_nonempty(self, default_config, brain_mri_ctx):
        recs = self._run_to_recommendations(brain_mri_ctx)
        assert len(recs) >= 1

    def test_last_recommendation_is_disclaimer(self, default_config, chest_ctx):
        recs = self._run_to_recommendations(chest_ctx)
        assert "licensed healthcare professional" in recs[-1]

    def test_no_recommendations_when_disabled(self):
        cfg = ClinicalConfiguration(include_recommendations=False)
        ctx = ClinicalContext(
            modality="skin", disease_label="melanoma", confidence_score=0.88
        )
        recs = self._run_to_recommendations(ctx, cfg)
        assert recs == []


# ── T07 — Full Pipeline (ClinicalExplanationEngine) ──────────────────────────

class TestClinicalExplanationEngine:

    def test_full_report_structure(self, engine, brain_mri_ctx):
        report = engine.explain(brain_mri_ctx)
        assert report["schema_version"] == "1.0.0"
        assert "clinical_impression" in report
        assert "findings" in report
        assert "severity" in report
        assert "recommendations" in report
        assert "performance" in report

    def test_report_confidence_matches_ctx(self, engine, brain_mri_ctx):
        report = engine.explain(brain_mri_ctx)
        assert abs(report["confidence_score"] - brain_mri_ctx.confidence_score) < 1e-4

    def test_full_pipeline_timing(self, engine, brain_mri_ctx):
        engine.explain(brain_mri_ctx)
        assert brain_mri_ctx.total_time_ms > 0

    def test_ctx_is_complete_after_explain(self, engine, brain_mri_ctx):
        engine.explain(brain_mri_ctx)
        assert brain_mri_ctx.is_complete

    def test_all_modalities_run_without_error(self, default_config):
        modalities = [
            ("brain_mri", "tumor"),
            ("chest_xray", "pneumonia"),
            ("brain_ct", "hemorrhage"),
            ("histopathology", "cancer"),
            ("retina", "diabetic_retinopathy"),
            ("skin", "melanoma"),
            ("ultrasound", "malignant"),
        ]
        eng = ClinicalExplanationEngine(config=default_config)
        for modality, disease in modalities:
            ctx = ClinicalContext(
                modality=modality,
                disease_label=disease,
                confidence_score=0.75,
                visualization_metrics={"coverage_ratio": 0.20},
            )
            report = eng.explain(ctx)
            assert report is not None
            assert ctx.is_complete
