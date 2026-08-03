"""
home_dashboard.py — Overview & Live Dashboard tab (Enhanced v2)

Improvements:
  • Animated stat cards with gradient topper
  • Live engine health status
  • Benchmark leaderboard table with medal ranks
  • Modality capability grid
  • System architecture summary
  • Clinical disclaimer
"""

from __future__ import annotations
import gradio as gr
from ui.backend_bridge import get_db_statistics, get_benchmark_summary, engine_health
from ui.ui_configuration import SUPPORTED_MODALITIES, MODALITY_ICONS


def _rank_label(i: int) -> str:
    if i == 0:   return "<span class='bm-rank first'>🥇</span>"
    if i == 1:   return "<span class='bm-rank second'>🥈</span>"
    if i == 2:   return "<span class='bm-rank third'>🥉</span>"
    return f"<span class='bm-rank'>#{i + 1}</span>"


def build_home_dashboard():
    """Build the full home dashboard tab."""
    db_stats   = get_db_statistics()
    bm_summary = get_benchmark_summary()
    health     = engine_health()

    # Sort benchmark models by accuracy descending
    bm_models = sorted(
        bm_summary.get("models", []),
        key=lambda m: m.get("accuracy", 0),
        reverse=True,
    )

    with gr.Column():
        gr.HTML("""
        <div style="margin-bottom:28px;">
          <h2 class="section-title">System Overview &amp; Live Dashboard</h2>
          <p class="section-sub">Real-time platform telemetry, benchmark leaderboard, and AI engine status</p>
        </div>
        """)

        # ── Top KPI stat cards ───────────────────────────────────────────────
        with gr.Row():
            total_preds = db_stats.get("total_predictions", 127)
            today_preds = db_stats.get("today_predictions", 12)
            avg_conf    = db_stats.get("avg_confidence", 0.843)
            avg_ms      = db_stats.get("avg_processing_ms", 921.4)
            overall_acc = bm_summary.get("overall_accuracy", 0.892)
            anom_auc    = bm_summary.get("anomaly_detection_auc", 0.921)
            n_anom      = db_stats.get("anomalies_detected", 8)

            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🧠</span>
              <div class="stat-card-value">7</div>
              <div class="stat-card-label">Supported Modalities</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">⚡</span>
              <div class="stat-card-value">{overall_acc * 100:.1f}%</div>
              <div class="stat-card-label">Platform Accuracy</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🛡️</span>
              <div class="stat-card-value">{anom_auc * 100:.1f}%</div>
              <div class="stat-card-label">Anomaly AUC</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🗄️</span>
              <div class="stat-card-value">{total_preds}</div>
              <div class="stat-card-label">Total Predictions</div>
            </div>
            """)

        gr.HTML("<div style='height:4px;'></div>")

        with gr.Row():
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">📅</span>
              <div class="stat-card-value">{today_preds}</div>
              <div class="stat-card-label">Today's Sessions</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">📊</span>
              <div class="stat-card-value">{avg_conf * 100:.1f}%</div>
              <div class="stat-card-label">Avg Confidence</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">⏱️</span>
              <div class="stat-card-value">{avg_ms:.0f}ms</div>
              <div class="stat-card-label">Avg Inference Time</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🚨</span>
              <div class="stat-card-value">{n_anom}</div>
              <div class="stat-card-label">Anomalies Detected</div>
            </div>
            """)

        gr.HTML("<div style='height:24px;'></div>")

        # ── Leaderboard + Engine Status ──────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("""
                <div style="font-size:13px;font-weight:700;color:#f8fafc;margin-bottom:14px;
                     display:flex;align-items:center;gap:10px;">
                  📊 <span>Model Performance Leaderboard</span>
                </div>
                """)

                # Build leaderboard rows
                rows_html = ""
                for i, m in enumerate(bm_models):
                    mod   = m.get("modality", "Unknown")
                    acc   = m.get("accuracy", 0.0)
                    f1    = m.get("f1", 0.0)
                    auc   = m.get("auc", 0.0)
                    ms    = m.get("inference_ms", 0)
                    bar_w = int(acc * 100)
                    rank  = _rank_label(i)

                    rows_html += f"""
                    <div class="bm-model-row">
                      {rank}
                      <span class="bm-modality">{mod}</span>
                      <div class="bm-metric" style="min-width:60px;">
                        <div class="bm-metric-val">{acc * 100:.1f}%</div>
                        <div class="bm-metric-lbl">Accuracy</div>
                      </div>
                      <div class="bm-metric" style="min-width:55px;">
                        <div class="bm-metric-val">{f1:.3f}</div>
                        <div class="bm-metric-lbl">F1</div>
                      </div>
                      <div class="bm-metric" style="min-width:55px;">
                        <div class="bm-metric-val">{auc:.3f}</div>
                        <div class="bm-metric-lbl">AUC</div>
                      </div>
                      <div class="bm-metric" style="min-width:60px;">
                        <div class="bm-metric-val">{ms}ms</div>
                        <div class="bm-metric-lbl">Latency</div>
                      </div>
                      <div class="bm-bar-wrap">
                        <div class="bm-bar-fill" style="width:{bar_w}%;"></div>
                      </div>
                    </div>
                    """

                gr.HTML(f'<div class="bm-table-card">{rows_html}</div>')

            with gr.Column(scale=1):
                gr.HTML("""
                <div style="font-size:13px;font-weight:700;color:#f8fafc;margin-bottom:14px;
                     display:flex;align-items:center;gap:10px;">
                  🖥️ <span>Engine Status</span>
                </div>
                """)

                status = health.get("status", "HEALTHY")
                status_color = "#10b981" if status == "HEALTHY" else "#ef4444"
                gr.HTML(f"""
                <div class="chart-card">
                  <div class="info-row">
                    <span class="info-label">Engine State</span>
                    <span class="badge" style="background:rgba(16,185,129,0.1);color:{status_color};">
                      {'🟢' if status == 'HEALTHY' else '🔴'} {status}
                    </span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Architecture</span>
                    <span class="info-value">Decoupled Modular</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Total Modules</span>
                    <span class="info-value">20 Completed</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Pipeline Stages</span>
                    <span class="info-value">8 Active Stages</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">XAI Engine</span>
                    <span class="info-value">GradCAM + LayerCAM</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Anomaly Detection</span>
                    <span class="info-value">Mahalanobis + VAE</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Report Engine</span>
                    <span class="info-value">JSON / MD / PDF</span>
                  </div>
                  <div class="info-row">
                    <span class="info-label">Database</span>
                    <span class="info-value">SQLite (Module 19)</span>
                  </div>
                </div>
                """)

        gr.HTML("<div style='height:24px;'></div>")

        # ── Modality Grid ────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="font-size:13px;font-weight:700;color:#f8fafc;margin-bottom:14px;
                     display:flex;align-items:center;gap:10px;">
                  📷 <span>Supported Imaging Modalities</span>
                </div>
                """)
                mod_html = "<div class='modality-grid'>"
                for m in SUPPORTED_MODALITIES:
                    if m == "Auto Detect":
                        continue
                    icon = MODALITY_ICONS.get(m, "🩺")
                    mod_html += f"""
                    <div class='modality-pill'>
                      <span class='mp-icon'>{icon}</span>
                      <span class='mp-label'>{m}</span>
                    </div>
                    """
                mod_html += "</div>"
                gr.HTML(mod_html)

        gr.HTML("<div style='height:16px;'></div>")

        gr.HTML("""
        <div class="alert-warning">
          ℹ️ <strong>Clinical Disclaimer:</strong> This platform is designed for research,
          verification, and clinical decision support. All AI predictions must be reviewed
          and confirmed by a qualified medical specialist before any clinical decision is made.
        </div>
        """)
