"""
benchmark_tab.py — Global Benchmarking & Model Leaderboard UI tab (Enhanced v2)
"""

from __future__ import annotations
import gradio as gr
from ui.backend_bridge import get_benchmark_summary


def _rank_badge(i: int) -> str:
    if i == 0: return "<span class='bm-rank first'>🥇</span>"
    if i == 1: return "<span class='bm-rank second'>🥈</span>"
    if i == 2: return "<span class='bm-rank third'>🥉</span>"
    return f"<span class='bm-rank'>#{i + 1}</span>"


def build_benchmark_tab():
    """Build global benchmarking and leaderboard tab with styled cards."""
    bm = get_benchmark_summary()
    models = sorted(bm.get("models", []), key=lambda m: m.get("accuracy", 0), reverse=True)

    with gr.Column():
        gr.HTML("""
        <div style="margin-bottom:24px;">
          <h2 class="section-title">📊 Global Model Leaderboard &amp; Benchmarking</h2>
          <p class="section-sub">Comprehensive performance evaluation across all 7 medical imaging modalities and specialized AI classifiers.</p>
        </div>
        """)

        # ── KPI Stat cards ───────────────────────────────────────────────────
        with gr.Row():
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🎯</span>
              <div class="stat-card-value">{bm.get('overall_accuracy', 0.892) * 100:.1f}%</div>
              <div class="stat-card-label">Mean Modality Accuracy</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">📈</span>
              <div class="stat-card-value">{bm.get('overall_auc', 0.934):.3f}</div>
              <div class="stat-card-label">Mean ROC-AUC</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🔍</span>
              <div class="stat-card-value">{bm.get('anomaly_detection_auc', 0.921):.3f}</div>
              <div class="stat-card-label">Anomaly Engine ROC-AUC</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🧪</span>
              <div class="stat-card-value">{bm.get('total_test_samples', 4200):,}</div>
              <div class="stat-card-label">Evaluated Test Scans</div>
            </div>
            """)

        gr.HTML("<div style='height:24px;'></div>")

        # ── Styled HTML Leaderboard Table ─────────────────────────────────────
        gr.HTML("""
        <div style="font-size:14px;font-weight:800;color:#f8fafc;margin-bottom:12px;
             display:flex;align-items:center;gap:10px;">
          🏆 <span>Modality Performance Leaderboard</span>
        </div>
        """)

        rows_html = ""
        for i, m in enumerate(models):
            mod   = m.get("modality", "Unknown")
            acc   = m.get("accuracy", 0.0)
            f1    = m.get("f1", 0.0)
            auc   = m.get("auc", 0.0)
            ms    = m.get("inference_ms", 0)
            bar_w = int(acc * 100)
            rank  = _rank_badge(i)

            acc_color = "#10b981" if acc >= 0.90 else ("#f59e0b" if acc >= 0.86 else "#38bdf8")

            rows_html += f"""
            <div class="bm-model-row">
              {rank}
              <span class="bm-modality">{mod}</span>
              <div class="bm-metric" style="min-width:70px;">
                <div class="bm-metric-val" style="color:{acc_color};">{acc * 100:.1f}%</div>
                <div class="bm-metric-lbl">Accuracy</div>
              </div>
              <div class="bm-metric" style="min-width:60px;">
                <div class="bm-metric-val">{f1 * 100:.1f}%</div>
                <div class="bm-metric-lbl">F1 Score</div>
              </div>
              <div class="bm-metric" style="min-width:60px;">
                <div class="bm-metric-val">{auc:.3f}</div>
                <div class="bm-metric-lbl">ROC-AUC</div>
              </div>
              <div class="bm-metric" style="min-width:70px;">
                <div class="bm-metric-val" style="font-family:'JetBrains Mono',monospace;">{ms}ms</div>
                <div class="bm-metric-lbl">Latency</div>
              </div>
              <div class="bm-bar-wrap">
                <div class="bm-bar-fill" style="width:{bar_w}%;background:linear-gradient(90deg, {acc_color}aa, {acc_color});"></div>
              </div>
            </div>
            """

        gr.HTML(f'<div class="bm-table-card">{rows_html}</div>')

        gr.HTML("<div style='height:24px;'></div>")

        # ── Detailed breakdown table (Dataframe) ──────────────────────────────
        gr.HTML("""
        <div style="font-size:13px;font-weight:700;color:#94a3b8;margin-bottom:8px;
             text-transform:uppercase;letter-spacing:1px;">
          Detailed Benchmark Metrics Table
        </div>
        """)

        df_rows = [
            [
                m["modality"],
                f"{m['accuracy'] * 100:.1f}%",
                f"{m['f1'] * 100:.1f}%",
                f"{m['auc']:.3f}",
                f"{m['inference_ms']} ms",
            ]
            for m in models
        ]

        gr.Dataframe(
            headers=["Imaging Modality", "Accuracy", "F1 Score", "ROC-AUC", "Mean Latency"],
            value=df_rows,
            interactive=False,
            elem_id="benchmark-leaderboard",
        )
