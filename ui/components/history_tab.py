"""
history_tab.py — Prediction History & Database UI tab (Enhanced)
"""

from __future__ import annotations
import gradio as gr
from ui.backend_bridge import get_prediction_history, get_db_statistics


def build_history_tab():
    """Build the Prediction History tab with searchable table and stats."""
    with gr.Column():
        gr.HTML("""
        <div style="margin-bottom:20px;">
          <h2 class="section-title">🗄️ Prediction History & Database</h2>
          <p class="section-sub">Browse, filter, and export all stored diagnostic sessions
          from the Module 19 SQLite database.</p>
        </div>
        """)

        # ── Filter controls ──────────────────────────────────────────────────
        with gr.Row():
            modality_filter = gr.Dropdown(
                choices=["All", "brain_mri", "chest_xray", "brain_ct",
                         "histopathology", "retina", "skin", "ultrasound"],
                value="All",
                label="Filter by Modality",
            )
            limit_slider = gr.Slider(minimum=5, maximum=200, value=50, step=5,
                                     label="Max Records")
            refresh_btn = gr.Button("🔄 Refresh History", variant="secondary",
                                    elem_classes=["btn-secondary"])

        # ── Stat summary row ─────────────────────────────────────────────────
        db_stats = get_db_statistics()

        with gr.Row():
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">📋</span>
              <div class="stat-card-value">{db_stats.get('total_predictions', 0)}</div>
              <div class="stat-card-label">Total Records</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">📊</span>
              <div class="stat-card-value">{db_stats.get('avg_confidence', 0) * 100:.1f}%</div>
              <div class="stat-card-label">Avg Confidence</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">⚡</span>
              <div class="stat-card-value">{db_stats.get('avg_processing_ms', 0):.0f}ms</div>
              <div class="stat-card-label">Avg Inference</div>
            </div>
            """)
            gr.HTML(f"""
            <div class="stat-card">
              <span class="stat-card-icon">🚨</span>
              <div class="stat-card-value">{db_stats.get('anomalies_detected', 0)}</div>
              <div class="stat-card-label">Anomalies Flagged</div>
            </div>
            """)

        gr.HTML("<div style='height:16px;'></div>")

        # ── History table ────────────────────────────────────────────────────
        history_output = gr.HTML(value="<div style='color:#94a3b8;'>Loading history...</div>")

        def render_history(mod_filter, limit):
            ok, records, msg = get_prediction_history(
                limit=int(limit),
                modality_filter=mod_filter if mod_filter != "All" else None,
            )
            if not records:
                return "<div style='color:#94a3b8;padding:24px;'>No records found.</div>"

            rows = ""
            for r in records:
                conf = float(r.get("confidence_score", 0.0))
                rel  = r.get("reliability", "low")
                rel_color = {"high": "#10b981", "moderate": "#f59e0b", "low": "#ef4444"}.get(rel.lower(), "#94a3b8")
                anom = "🚨 Yes" if r.get("is_anomaly") else "✅ No"
                anom_col = "#ef4444" if r.get("is_anomaly") else "#10b981"
                rows += f"""
                <tr>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#60a5fa;">
                    #{str(r.get('request_id', 'N/A'))[:8]}
                  </td>
                  <td>{r.get('timestamp', 'N/A')[:19].replace('T',' ')}</td>
                  <td>{r.get('patient_id', 'N/A')}</td>
                  <td><span class="badge badge-info">{r.get('modality', 'N/A').replace('_',' ').title()}</span></td>
                  <td style="font-weight:700;color:#f8fafc;">{r.get('predicted_class', 'N/A')}</td>
                  <td style="font-weight:700;color:{rel_color};">{conf * 100:.1f}%</td>
                  <td><span class="badge" style="color:{rel_color};background:rgba(16,185,129,0.08);">{rel.upper()}</span></td>
                  <td style="font-weight:700;color:{anom_col};">{anom}</td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:12px;">{r.get('total_ms', 0):.0f}ms</td>
                </tr>
                """

            header = """
            <tr>
              <th>Case ID</th><th>Timestamp</th><th>Patient</th>
              <th>Modality</th><th>Diagnosis</th><th>Confidence</th>
              <th>Reliability</th><th>Anomaly</th><th>Latency</th>
            </tr>
            """
            info = f"<div style='color:#94a3b8;font-size:11px;padding:8px 0;'>Showing {len(records)} records</div>"
            if msg:
                info += f"<div class='alert-warning' style='margin-bottom:8px;'>{msg}</div>"

            return info + f"<div class='bm-table-card'><table>{header}{rows}</table></div>"

        # Initial load
        history_output.value = render_history("All", 50)

        refresh_btn.click(
            fn=render_history,
            inputs=[modality_filter, limit_slider],
            outputs=[history_output],
        )
        modality_filter.change(
            fn=render_history,
            inputs=[modality_filter, limit_slider],
            outputs=[history_output],
        )
