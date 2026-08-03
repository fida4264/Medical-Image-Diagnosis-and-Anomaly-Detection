"""
about_tab.py — Platform Architecture & Documentation UI tab
"""
import gradio as gr
from ui.ui_configuration import (
    PLATFORM_NAME,
    PLATFORM_VERSION,
    APP_DESCRIPTION,
    DEVELOPER_NAME,
    INSTITUTION,
    DEPARTMENT,
    TECH_STACK,
)


def build_about_tab():
    """Build about page displaying platform architecture, technology stack, and developer credits."""
    with gr.Column():
        gr.HTML(
            f"""
            <div class="about-hero">
                <div class="about-hero-title">{PLATFORM_NAME}</div>
                <div class="about-hero-sub">Production-Grade Medical AI Engine & Anomaly Detection Framework</div>
                <div style="margin-top: 12px;">
                    <span class="badge badge-info">{PLATFORM_VERSION}</span>
                    <span class="badge badge-success">Architecture Locked</span>
                    <span class="badge badge-low">20 Modules Verified</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📖 Executive Overview")
                gr.Markdown(APP_DESCRIPTION)

                gr.Markdown("### 🏗️ Core Architecture Principles")
                gr.Markdown(
                    """
                    1. **Strict Decoupling**: Complete isolation between web interface, orchestration engine (`MedicalAIEngine`), and individual disease models.
                    2. **Single State Carrier**: Every pipeline stage receives and mutates a single `EngineContext` object.
                    3. **Universal Adapter Pattern**: Unified predictor interface (`BasePredictorAdapter`) normalizing all 7 modalities.
                    4. **Explainability First**: Native integration of Layer-CAM heatmap visual explanation.
                    5. **Anomaly Safety Net**: Out-of-distribution Mahalanobis feature distance and confidence thresholding for unknown conditions.
                    """
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Technology Stack")
                stack_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
                for name, icon in TECH_STACK:
                    stack_html += f"""
                    <span class="badge badge-info" style="font-size: 13px; padding: 6px 14px;">
                        {icon} {name}
                    </span>
                    """
                stack_html += "</div>"
                gr.HTML(stack_html)

                gr.HTML("<div style='height: 20px;'></div>")

                gr.Markdown("### 👨‍💻 Developer & Project Credits")
                gr.HTML(
                    f"""
                    <div class="chart-card">
                        <div class="info-row">
                            <span class="info-label">Developer</span>
                            <span class="info-value">{DEVELOPER_NAME}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Department</span>
                            <span class="info-value">{DEPARTMENT}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Institution</span>
                            <span class="info-value">{INSTITUTION}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">License</span>
                            <span class="info-value">MIT / Academic Research</span>
                        </div>
                    </div>
                    """
                )
