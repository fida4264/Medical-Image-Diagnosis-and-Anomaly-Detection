# Module 16.5 — Risk Assessment & Decision Fusion Subsystem

**Medical Image Diagnosis and Anomaly Detection Platform**  
**Phase 5 — Advanced AI & Anomaly Detection**  
**Status:** ✅ COMPLETE · 106/106 Anomaly Tests Passed (281 Total Platform Tests Passed)

---

## 1. Overview

Module 16.5 implements the **Risk Assessment & Decision Fusion** subsystem (`medical_ai_engine/anomaly/decision/`). It receives multiple upstream anomaly detection signals and synthesizes them into a single, authoritative `AnomalyResult` with a unified clinical risk decision.

---

## 2. Fusion Architecture

```
Input Signals:
 - Confidence Threshold Result
 - Mahalanobis OOD Result
 - Unknown Disease Result
 - Image Quality Check Result
         │
         ▼
 1. PriorityManager         ──── Signal priority resolution & hierarchy
         │
         ▼
 2. DecisionPolicy          ──── DefaultPriorityPolicy or WeightedFusionPolicy
         │
         ▼
 3. FusionEngine            ──── Fuses signals into intermediate DecisionOutput
         │
         ▼
 4. RiskAssessor            ──── Assigns Clinical Risk Level (LOW/MODERATE/HIGH/CRITICAL)
         │
         ▼
 5. DecisionEngine          ──── Constructs final immutable AnomalyResult
```

---

## 3. Package Structure

```
medical_ai_engine/
└── anomaly/
    └── decision/
        ├── __init__.py                  ← Package exports
        ├── decision_configuration.py    ← DecisionConfiguration
        ├── decision_exceptions.py       ← Typed exception hierarchy
        ├── priority_manager.py          ← PriorityManager & SignalType
        ├── decision_policy.py           ← IDecisionPolicy, DefaultPriorityPolicy, WeightedFusionPolicy
        ├── fusion_engine.py             ← FusionEngine
        ├── risk_assessor.py             ├── RiskAssessor & RiskAssessmentOutput
        └── decision_engine.py           ← DecisionEngine master orchestrator
```

---

## 4. Priority Rules & Policies

### Default Priority Policy (`default_priority`)
Evaluates signals in priority order:
1. **Unknown Disease** (Rank 1) → `ESCALATE` / `HIGH` Risk
2. **Out of Distribution (OOD)** (Rank 2) → `ESCALATE` / `HIGH` Risk
3. **Low Confidence** (Rank 3) → `MANUAL_REVIEW` or `ESCALATE` / `MODERATE`–`HIGH` Risk
4. **Poor Image Quality** (Rank 4) → `REPEAT_SCAN` / `MODERATE` Risk
5. **Known Disease** (Rank 5 — Baseline) → `PROCEED` / `LOW` Risk

### Weighted Fusion Policy (`weighted_fusion`)
Fuses numeric risk scores:
$$R_{\text{composite}} = w_{\text{unk}} S_{\text{unk}} + w_{\text{ood}} S_{\text{ood}} + w_{\text{conf}} (1 - \text{confidence})$$
Threshold mapping:
- $R_{\text{composite}} \ge \text{CRITICAL\_THRESHOLD} \, (0.88)$ $\rightarrow$ `CRITICAL` risk / `REJECT_PREDICTION`
- $R_{\text{composite}} \ge \text{HIGH\_THRESHOLD} \, (0.70)$ $\rightarrow$ `HIGH` risk / `ESCALATE`
- $R_{\text{composite}} \ge 0.40$ $\rightarrow$ `MODERATE` risk / `MANUAL_REVIEW`
- $R_{\text{composite}} < 0.40$ $\rightarrow$ `LOW` risk / `PROCEED`

---

## 5. Output AnomalyResult Schema

- `status`: `AnomalyStatus` enum (`KNOWN_DISEASE`, `UNKNOWN_DISEASE`, `OUT_OF_DISTRIBUTION`, `LOW_CONFIDENCE`, `POOR_QUALITY`)
- `risk_level`: `RiskLevel` enum (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`)
- `known_unknown_status`: `str` (`"KNOWN"`, `"UNKNOWN"`, `"UNCERTAIN"`)
- `risk_score`: `float` in $[0.0, 1.0]$
- `recommended_action`: `RecommendedAction` enum (`PROCEED`, `MANUAL_REVIEW`, `REPEAT_SCAN`, `ESCALATE`, `REJECT_PREDICTION`)
- `reason`: Human-readable clinical rationale string
- `metadata`: Contains `decision_confidence`, `primary_signal`, and `policy_name`

---

## 6. Extension Points

1. **Custom Decision Policies**: Implement `IDecisionPolicy` interface and register via `PolicyRegistry.register("custom_name", CustomPolicyCls)`.
2. **Custom Priority Hierarchies**: Instantiate `PriorityManager` with custom signal lists.
3. **Per-Modality Threshold Overrides**: Configure `DecisionConfiguration.per_modality_thresholds`.

---

## 7. Verification Results

```
============================ 106 passed in 0.78s =============================
(106 anomaly tests + 175 explainability tests = 281 total platform tests)
```
