# PROBLEM_STATEMENT.md — Research Problem Statement

---

## 1. Clinical Context & Problem Motivation

Medical imaging is the cornerstone of modern diagnostic medicine, enabling non-invasive visualization of internal anatomical structures and pathological changes. Despite dramatic advances in imaging technology—MRI, CT, fundus photography, dermoscopy, digital pathology, and ultrasound—the interpretation of these images remains almost exclusively dependent on specialist clinicians whose availability is globally unequal.

The World Health Organization (WHO) estimates a global shortage of 4.3 million health workers, disproportionately affecting low- and middle-income countries where specialist radiologists, pathologists, and ophthalmologists are critically understaffed. Even in well-resourced environments, the exponential growth of imaging volume—driven by population growth, aging demographics, and expanded screening programs—creates systemic diagnostic backlogs that delay treatment initiation.

---

## 2. Research Gaps Addressed

Existing approaches to automated medical image diagnosis fall into several categories, each with identified limitations that this platform addresses:

### Gap 1: Single-Modality Silos
The majority of published AI diagnostic systems are designed for a single imaging modality (e.g., chest X-ray only, or skin lesion only). Clinical practice requires simultaneous competence across modalities, as patients present with diverse imaging types.

**This platform unifies 7 imaging modalities** under a single orchestrated AI framework.

### Gap 2: Lack of Explainability
Black-box neural network predictions without visual evidence are unsuitable for clinical acceptance or regulatory compliance. Clinicians require visual justification for AI recommendations.

**This platform integrates GradCAM explainability** producing anatomically localized heatmaps that highlight disease-relevant regions.

### Gap 3: No Out-of-Distribution Detection
Models trained on specific disease distributions fail silently on novel pathology, image artefacts, or mismatched scanner protocols—returning high-confidence but incorrect predictions.

**This platform implements a dedicated OOD Anomaly Detection Engine** flagging inputs that deviate from training distribution.

### Gap 4: No Production Deployment Path
Academic AI models are rarely packaged for real-world deployment. The gap between research prototype and deployable system is substantial.

**This platform delivers production-grade REST API, containerized Docker images, and Kubernetes manifests** for direct hospital system integration.

### Gap 5: Absence of Audit & Governance Infrastructure
Clinical AI systems require prediction history, audit trails, and structured report generation for regulatory compliance (HIPAA, GDPR, ISO 13485).

**This platform implements SQLite prediction audit logging, medical report generation (PDF/MD/JSON), and structured API authentication.**

---

## 3. Research Questions

1. Can a single unified deep learning framework effectively classify pathological conditions across 7 distinct medical imaging modalities?
2. Does GradCAM-based visual explainability provide clinically interpretable localization of disease-relevant features?
3. Can OOD anomaly detection reliably identify out-of-distribution inputs before they produce misleading diagnostic predictions?
4. Is a production-grade containerized medical AI platform deployable without manual environment configuration?
