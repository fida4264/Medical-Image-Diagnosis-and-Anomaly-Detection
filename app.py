"""
app.py — Module 20: Professional Gradio Web Application
Medical Image Diagnosis and Anomaly Detection Platform

Entry point for launching the Gradio web dashboard.

Usage:
    python app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import gradio as gr

# Ensure root is on path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.theme import MEDICAL_CSS, get_theme
from ui.ui_configuration import PLATFORM_NAME, PLATFORM_VERSION
from ui.components.header import build_header
from ui.components.landing_page import build_landing_page
from ui.components.demo_mode import build_demo_mode_section, run_demo_pipeline
from ui.components.home_dashboard import build_home_dashboard
from ui.components.prediction_tab import build_prediction_tab
from ui.components.history_tab import build_history_tab
from ui.components.benchmark_tab import build_benchmark_tab
from ui.components.about_tab import build_about_tab
from ui.components.footer import build_footer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalAI.App")


def create_app() -> gr.Blocks:
    """Construct and configure the Gradio Blocks application."""
    theme = get_theme()

    with gr.Blocks(
        title=f"{PLATFORM_NAME} {PLATFORM_VERSION}",
        css=MEDICAL_CSS,
        theme=theme,
    ) as app:

        # ── Sticky Header Banner ───────────────────────────────────────────────
        build_header()

        # ── Landing Overlay Screen ─────────────────────────────────────────────
        landing_col, launch_btn, demo_btn = build_landing_page()

        # ── Navigation Tabs ────────────────────────────────────────────────────
        with gr.Tabs() as main_tabs:

            with gr.TabItem("🏠 Overview & Dashboard", id="tab_home"):
                build_home_dashboard()

            with gr.TabItem("🔬 Diagnostic Analysis", id="tab_predict"):
                build_prediction_tab()

            with gr.TabItem("🎬 Automated Demo", id="tab_demo"):
                mod_radio, run_demo_btn, demo_status = build_demo_mode_section()

                def handle_demo_click(modality_val):
                    from ui.components.prediction_tab import (
                        _confidence_gauge, _risk_badge, _topk_bars
                    )

                    success, res, err, img_path = run_demo_pipeline(modality_val)
                    if not success:
                        return f"<div class='alert-error'>❌ Demo Execution Failed: {err}</div>"

                    disease      = res.get("predicted_class", "Unknown")
                    conf         = float(res.get("confidence_score", 0.0))
                    reliability  = res.get("reliability", "low")
                    req_review   = bool(res.get("requires_review", True))
                    total_ms     = float(res.get("total_ms", 0.0))
                    mod_display  = res.get("modality", "Unknown").replace("_", " ").title()
                    gauge_html   = _confidence_gauge(conf, reliability)
                    rbadge       = _risk_badge(reliability, req_review)
                    topk_html    = _topk_bars(res.get("top_predictions", []))
                    request_id   = res.get("request_id", "")[:8]

                    anom_score   = float(res.get("anomaly_score", 0.0))
                    is_anom      = bool(res.get("is_anomaly", False))
                    anom_color   = "#ef4444" if is_anom else "#10b981"
                    anom_label   = "OOD ANOMALY DETECTED" if is_anom else "NORMAL IN-DISTRIBUTION SCAN"
                    anom_cls     = "high-risk" if is_anom else "normal"

                    return f"""
                    <div class="alert-success" style="margin-bottom:16px;">
                      ✅ <strong>Automated Demo Completed!</strong> Processed synthetic <strong>{mod_display}</strong> scan through all 8 pipeline stages in <strong>{total_ms:.1f} ms</strong>.
                    </div>

                    <div class="result-card">
                      <div class="result-card-header">
                        <span style="font-size:22px;">🎬</span>
                        <span class="result-card-title">Demo Analysis Result — Case #{request_id}</span>
                        <span style="margin-left:auto;">{rbadge}</span>
                      </div>

                      <div style="display:flex; gap:20px; align-items:flex-start; margin:16px 0;">
                        <div style="flex:1;">
                          <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;">Predicted Condition</div>
                          <div class="disease-name" style="margin-top:4px;">{disease}</div>
                          <div style="margin-top:16px;">
                            <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;margin-bottom:8px;">Class Probabilities</div>
                            {topk_html}
                          </div>
                        </div>
                        <div style="flex-shrink:0;">{gauge_html}</div>
                      </div>

                      <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                        <div class="info-row"><span class="info-label">Imaging Modality</span><span class="info-value">{mod_display}</span></div>
                        <div class="info-row"><span class="info-label">Anomaly Status</span><span class="badge {anom_cls}">{anom_label} ({anom_score:.3f})</span></div>
                        <div class="info-row"><span class="info-label">Pipeline Latency</span><span class="info-value" style="font-family:'JetBrains Mono',monospace;">{total_ms:.1f} ms</span></div>
                        <div class="info-row"><span class="info-label">Pipeline Status</span><span class="badge badge-success">8/8 Stages Passed</span></div>
                      </div>
                    </div>
                    """

                run_demo_btn.click(
                    fn=handle_demo_click,
                    inputs=[mod_radio],
                    outputs=[demo_status],
                )

            with gr.TabItem("🗄️ Prediction History", id="tab_history"):
                build_history_tab()

            with gr.TabItem("📊 Global Benchmark", id="tab_benchmark"):
                build_benchmark_tab()

            with gr.TabItem("ℹ️ About Architecture", id="tab_about"):
                build_about_tab()

        # ── Footer Banner ──────────────────────────────────────────────────────
        build_footer()

        # ── Landing Overlay Event Handlers ─────────────────────────────────────
        def launch_app():
            return gr.update(visible=False)

        def launch_demo():
            return gr.update(visible=False)

        launch_btn.click(fn=launch_app, outputs=[landing_col])
        demo_btn.click(fn=launch_demo, outputs=[landing_col])

    return app


if __name__ == "__main__":
    logger.info("Initializing Medical AI Web Application (Module 20)...")
    application = create_app()
    logger.info("Launching server on http://127.0.0.1:7860 ...")
    application.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
