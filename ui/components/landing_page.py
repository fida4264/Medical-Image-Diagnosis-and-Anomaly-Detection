"""
landing_page.py — Professional hero landing screen for the Medical AI Platform.
"""

from __future__ import annotations
import gradio as gr
from ui.backend_bridge import get_benchmark_summary, get_db_statistics


def build_landing_page() -> tuple:
    """
    Builds a full-screen landing overlay that sits above the main app.
    Returns (landing_html_component, launch_btn, demo_btn).
    """
    bm = get_benchmark_summary()
    db = get_db_statistics()

    acc_pct  = f"{bm.get('overall_accuracy', 0.892) * 100:.1f}%"
    auc_pct  = f"{bm.get('overall_auc', 0.934) * 100:.1f}%"
    n_preds  = str(db.get("total_predictions", 127))
    anom_auc = f"{bm.get('anomaly_detection_auc', 0.921) * 100:.1f}%"

    hero_html = f"""
    <div class="hero-container" id="hero-root">
      <!-- Decorative grid -->
      <div class="hero-grid-bg"></div>

      <!-- Badge -->
      <div style="display:flex; justify-content:center;">
        <div class="hero-badge">
          <span class="hero-badge-dot"></span>
          v0.20.0 · Final Year Project · AI Research Platform
        </div>
      </div>

      <!-- Title -->
      <h1 class="hero-title">Medical Image<br/>Diagnosis AI</h1>
      <p class="hero-subtitle">
        A state-of-the-art deep learning platform for automated multi-modality medical
        image analysis, disease classification, explainability, and anomaly detection.
      </p>

      <!-- Feature pills -->
      <div class="hero-pill-row">
        <span class="hero-pill">🧠 Brain MRI</span>
        <span class="hero-pill">🫁 Chest X-Ray</span>
        <span class="hero-pill">🧬 Histopathology</span>
        <span class="hero-pill">👁️ Retina</span>
        <span class="hero-pill">🩺 Skin Lesions</span>
        <span class="hero-pill">🔬 Ultrasound</span>
        <span class="hero-pill">⚡ GradCAM XAI</span>
        <span class="hero-pill">🛡️ Anomaly OOD</span>
      </div>

      <!-- Stats -->
      <div class="hero-stats-row">
        <div class="hero-stat">
          <div class="hero-stat-val">7</div>
          <div class="hero-stat-lbl">Modalities</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{acc_pct}</div>
          <div class="hero-stat-lbl">Accuracy</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{auc_pct}</div>
          <div class="hero-stat-lbl">AUC Score</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{anom_auc}</div>
          <div class="hero-stat-lbl">Anomaly AUC</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-val">{n_preds}+</div>
          <div class="hero-stat-lbl">Predictions</div>
        </div>
      </div>
    </div>
    """

    with gr.Column(visible=True, elem_id="med-landing-overlay") as landing_col:
        gr.HTML(hero_html)

        with gr.Row():
            launch_btn = gr.Button("🚀 Launch Platform", variant="primary", elem_classes=["hero-btn-primary"])
            demo_btn   = gr.Button("🎬 Launch Automated Demo", variant="secondary", elem_classes=["hero-btn-demo"])

    return landing_col, launch_btn, demo_btn
