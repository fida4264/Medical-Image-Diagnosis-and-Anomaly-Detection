"""
ui_configuration.py — Module 20: Professional Gradio Web Application
Medical AI Platform — UI settings and constants.
"""

from __future__ import annotations

PLATFORM_NAME = "Medical AI Diagnostic Platform"
PLATFORM_SUBTITLE = "Intelligent Medical Image Analysis & Anomaly Detection System"
PLATFORM_VERSION = "v0.20.0.0"
DEVELOPER_NAME = "Fidah"
INSTITUTION = "University"
DEPARTMENT = "Computer Science & Artificial Intelligence"

SUPPORTED_MODALITIES = [
    "Auto Detect",
    "Brain MRI",
    "Chest X-Ray",
    "Brain CT",
    "Histopathology",
    "Retina",
    "Skin Lesion",
    "Ultrasound",
]

MODALITY_KEY_MAP = {
    "Auto Detect": None,
    "Brain MRI": "brain_mri",
    "Chest X-Ray": "chest_xray",
    "Brain CT": "brain_ct",
    "Histopathology": "histopathology",
    "Retina": "retina",
    "Skin Lesion": "skin",
    "Ultrasound": "ultrasound",
}

MODALITY_ICONS = {
    "Auto Detect": "🔍",
    "Brain MRI": "🧠",
    "Chest X-Ray": "🫁",
    "Brain CT": "💡",
    "Histopathology": "🔬",
    "Retina": "👁️",
    "Skin Lesion": "🩺",
    "Ultrasound": "📡",
}

RISK_COLORS = {
    "LOW": "#10b981",
    "MODERATE": "#f59e0b",
    "HIGH": "#ef4444",
    "CRITICAL": "#7c3aed",
}

RISK_ICONS = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🔴",
    "CRITICAL": "🔴",
}

ANOMALY_LABELS = {
    "KNOWN_DISEASE": ("Known Disease", "#10b981"),
    "UNKNOWN_DISEASE": ("Unknown Pattern", "#f59e0b"),
    "OUT_OF_DISTRIBUTION": ("Out-of-Distribution", "#ef4444"),
    "LOW_CONFIDENCE": ("Low Confidence", "#6366f1"),
}

TECH_STACK = [
    ("Python 3.12", "🐍"),
    ("PyTorch 2.x", "🔥"),
    ("OpenCV", "📷"),
    ("Gradio 4.x", "🎛️"),
    ("SQLite", "🗄️"),
    ("NumPy / SciPy", "🔢"),
    ("Pillow (PIL)", "🖼️"),
]

APP_DESCRIPTION = (
    "A production-grade Medical Image Diagnosis & Anomaly Detection Platform powered by "
    "deep learning. Supports 7 imaging modalities with disease classification, GradCAM "
    "visual explainability, out-of-distribution anomaly detection, automated clinical "
    "report generation, and comprehensive prediction history management."
)

__all__ = [
    "PLATFORM_NAME",
    "PLATFORM_SUBTITLE",
    "PLATFORM_VERSION",
    "DEVELOPER_NAME",
    "INSTITUTION",
    "DEPARTMENT",
    "SUPPORTED_MODALITIES",
    "MODALITY_KEY_MAP",
    "MODALITY_ICONS",
    "RISK_COLORS",
    "RISK_ICONS",
    "ANOMALY_LABELS",
    "TECH_STACK",
    "APP_DESCRIPTION",
]
