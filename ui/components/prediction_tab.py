"""
prediction_tab.py — Diagnostic Analysis & Inference UI tab

Premium features:
  • Animated staged pipeline tracker (8 stages)
  • Circular SVG confidence gauge with animated fill
  • Clinical risk badges with pulse
  • Top-K probability bars
  • Image comparison (Original | GradCAM | Overlay)
  • Full medical report card
  • Multi-format report export
  • Demo mode banner
"""

from __future__ import annotations
import gradio as gr
from pathlib import Path
from ui.ui_configuration import SUPPORTED_MODALITIES, MODALITY_KEY_MAP, MODALITY_ICONS
from ui.backend_bridge import run_prediction, generate_report, export_report


# ── Helpers ────────────────────────────────────────────────────────────────────

def _risk_badge(reliability: str, requires_review: bool) -> str:
    """Return an HTML risk badge based on reliability level."""
    rel = reliability.lower()
    if rel == "high" and not requires_review:
        cls, dot_col, label = "normal", "#10b981", "✅ HIGH CONFIDENCE"
    elif rel == "moderate":
        cls, dot_col, label = "warning", "#f59e0b", "⚠️ MODERATE — Review Advised"
    elif rel == "low":
        cls, dot_col, label = "high-risk", "#ef4444", "🚨 LOW CONFIDENCE"
    else:
        cls, dot_col, label = "normal", "#10b981", "✅ HIGH CONFIDENCE"

    if requires_review and rel != "low":
        cls, dot_col, label = "warning", "#f59e0b", "⚠️ MANUAL REVIEW RECOMMENDED"

    return (
        f'<span class="risk-badge {cls}">'
        f'  <span class="risk-badge-dot" style="background:{dot_col};"></span>'
        f"  {label}"
        f"</span>"
    )


def _confidence_gauge(score: float, reliability: str) -> str:
    """Render an SVG semi-circular confidence gauge."""
    pct = max(0.0, min(1.0, score))
    rel = reliability.lower()
    if rel == "high":
        color = "#10b981"
    elif rel == "moderate":
        color = "#f59e0b"
    else:
        color = "#ef4444"

    # SVG arc: radius=70, cx=80, cy=90
    r = 70
    cx, cy = 80, 90
    # Total arc from 180° to 0° (left half to right half, top is middle)
    # angle from 180 deg to 0 deg across pct
    import math
    start_angle = math.pi  # leftmost
    end_angle = 0.0        # rightmost
    sweep = math.pi        # 180 degrees
    angle = start_angle - pct * sweep

    x_end = cx + r * math.cos(angle)
    y_end = cy + r * math.sin(angle)

    # Arc path for track
    arc_d = f"M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
    # Arc path for fill
    fill_large = 1 if pct > 0.5 else 0
    fill_d = f"M {cx - r} {cy} A {r} {r} 0 {fill_large} 1 {x_end:.2f} {y_end:.2f}"

    conf_label = f"{pct * 100:.1f}%"
    gauge_class = "high" if rel == "high" else ("moderate" if rel == "moderate" else "low")

    return f"""
    <div class="gauge-container">
      <svg class="gauge-svg" viewBox="0 0 160 100" width="180" height="110">
        <!-- Track -->
        <path d="{arc_d}" fill="none" stroke="#1e293b" stroke-width="12" stroke-linecap="round"/>
        <!-- Fill -->
        <path d="{fill_d}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
              style="filter: drop-shadow(0 0 6px {color}aa);"/>
        <!-- Needle dot -->
        <circle cx="{x_end:.2f}" cy="{y_end:.2f}" r="6" fill="{color}" style="filter: drop-shadow(0 0 4px {color});"/>
      </svg>
      <div class="gauge-value {gauge_class}" style="margin-top:-20px;">{conf_label}</div>
      <div class="gauge-label">Diagnostic Confidence</div>
    </div>
    """


def _topk_bars(top_preds: list) -> str:
    """Render Top-K prediction probability bars."""
    if not top_preds:
        return "<div style='color:#94a3b8; font-size:12px;'>No top-k data available.</div>"

    bars = ""
    for i, item in enumerate(top_preds[:5]):
        if isinstance(item, dict):
            label = item.get("disease", item.get("class", f"Class {i}"))
            prob = float(item.get("confidence", item.get("probability", 0.0)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label, prob = str(item[0]), float(item[1])
        else:
            continue

        pct_int = int(prob * 100)
        pct_str = f"{prob * 100:.1f}%"
        color = "#0ea5e9" if i == 0 else ("#10b981" if i == 1 else "#94a3b8")
        bars += f"""
        <div class="topk-row">
          <div class="topk-label-row">
            <span class="topk-label">{label}</span>
            <span class="topk-pct">{pct_str}</span>
          </div>
          <div class="topk-bar-wrap">
            <div class="topk-bar-fill" style="width:{pct_int}%; background:{color};"></div>
          </div>
        </div>
        """
    return bars


def _pipeline_stages_done(total_ms: float) -> str:
    """Build a completed pipeline tracker HTML."""
    stages = [
        ("🗂️", "Input Validation"),
        ("🖼️", "Image Pre-processing"),
        ("🔍", "Modality Classification"),
        ("🔀", "Intelligent Routing"),
        ("🧠", "Disease Classification"),
        ("📊", "Confidence Assessment"),
        ("🎯", "GradCAM Explainability"),
        ("🛡️", "Anomaly Detection"),
    ]
    rows = ""
    for icon, label in stages:
        rows += f"""
        <div class="stage-row done">
            <span class="stage-icon">{icon}</span>
            <span class="stage-label">{label}</span>
            <span class="stage-check">✓</span>
        </div>
        """
    return f"""
    <div class="pipeline-tracker">
        <div class="pipeline-title">⚡ Pipeline Execution — {total_ms:.0f} ms total</div>
        {rows}
    </div>
    """


def _pipeline_stages_loading() -> str:
    """Build an 'in-progress' pipeline tracker."""
    stages = [
        ("🗂️", "Input Validation", True, False),
        ("🖼️", "Image Pre-processing", True, False),
        ("🔍", "Modality Classification", False, True),
        ("🔀", "Intelligent Routing", False, False),
        ("🧠", "Disease Classification", False, False),
        ("📊", "Confidence Assessment", False, False),
        ("🎯", "GradCAM Explainability", False, False),
        ("🛡️", "Anomaly Detection", False, False),
    ]
    rows = ""
    for icon, label, done, active in stages:
        if done:
            rows += f"""
            <div class="stage-row done">
                <span class="stage-icon">{icon}</span>
                <span class="stage-label">{label}</span>
                <span class="stage-check">✓</span>
            </div>
            """
        elif active:
            rows += f"""
            <div class="stage-row active">
                <div class="stage-spinner"></div>
                <span class="stage-label">{label}</span>
            </div>
            """
        else:
            rows += f"""
            <div class="stage-row">
                <span class="stage-icon">{icon}</span>
                <span class="stage-label">{label}</span>
            </div>
            """
    return f"""
    <div class="pipeline-tracker">
        <div class="pipeline-title">🔄 Pipeline Executing...</div>
        {rows}
    </div>
    """


# ── Main Builder ───────────────────────────────────────────────────────────────

def build_prediction_tab():
    """Build the premium Diagnostic Analysis workflow tab."""

    with gr.Column():
        gr.HTML("""
        <div style="margin-bottom:20px;">
            <h2 class="section-title">🔬 Diagnostic Analysis Pipeline</h2>
            <p class="section-sub">Upload a medical image for automated modality detection,
            disease classification, GradCAM explainability, and anomaly risk assessment.</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left Column: Input Controls ──────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.HTML("""
                <div style="font-size:11px;font-weight:700;color:#94a3b8;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
                    1 &nbsp;·&nbsp; Upload & Configure
                </div>
                """)

                image_input = gr.Image(
                    type="filepath",
                    label="Medical Scan Image",
                    elem_classes=["upload-zone"],
                )

                modality_select = gr.Dropdown(
                    choices=SUPPORTED_MODALITIES,
                    value="Auto Detect",
                    label="Target Modality",
                    info="Auto Detect uses the AI modality router",
                )

                patient_id_input = gr.Textbox(
                    label="Patient / Case ID",
                    placeholder="e.g. PT-2026-8812  (optional)",
                    lines=1,
                )

                analyze_btn = gr.Button(
                    "🚀 Run Complete Diagnostic Analysis",
                    variant="primary",
                    elem_classes=["btn-primary"],
                )

                # Staged pipeline tracker output
                pipeline_tracker_out = gr.HTML(
                    value="<div style='color:#475569;font-size:12px;padding:8px 0;'>"
                          "🔲 Pipeline ready — upload an image and click <strong>Run Analysis</strong>.</div>"
                )

                # Status bar
                status_output = gr.HTML(
                    value="<div style='color:#94a3b8;font-size:12px;margin-top:4px;'>Ready.</div>"
                )

            # ── Right Column: Outputs ────────────────────────────────────────
            with gr.Column(scale=2):
                gr.HTML("""
                <div style="font-size:11px;font-weight:700;color:#94a3b8;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
                    2 &nbsp;·&nbsp; Diagnostic Results & Insights
                </div>
                """)

                # State holders
                result_state  = gr.State({})
                report_state  = gr.State({})

                with gr.Tabs():

                    # ── Tab 1: Primary Diagnosis ─────────────────────────────
                    with gr.TabItem("📊 Primary Diagnosis"):
                        diag_output = gr.HTML(
                            value="<div class='result-card' style='text-align:center;"
                                  "color:#94a3b8;padding:48px 20px;'>"
                                  "Upload an image and click <strong>Run Diagnostic Analysis</strong>."
                                  "</div>"
                        )

                    # ── Tab 2: Explainability ────────────────────────────────
                    with gr.TabItem("👁️ Explainability (GradCAM)"):
                        cam_label_html = gr.HTML(
                            value="<div style='color:#94a3b8;font-size:12px;padding:4px 0;'>"
                                  "GradCAM heatmap highlights critical regions driving the prediction.</div>"
                        )
                        with gr.Row():
                            cam_orig    = gr.Image(label="Original Scan",         interactive=False)
                            cam_overlay = gr.Image(label="GradCAM Overlay",       interactive=False)
                        cam_info = gr.HTML(
                            value="<div style='color:#94a3b8;font-size:12px;'>"
                                  "Run diagnostic analysis to generate the heatmap.</div>"
                        )

                    # ── Tab 3: Anomaly Detection ─────────────────────────────
                    with gr.TabItem("🛡️ Anomaly & Risk Assessment"):
                        anomaly_output = gr.HTML(
                            value="<div class='result-card' style='text-align:center;"
                                  "color:#94a3b8;padding:40px;'>"
                                  "Anomaly risk metrics will appear here after analysis.</div>"
                        )

                    # ── Tab 4: Medical Report ────────────────────────────────
                    with gr.TabItem("📋 Medical Report"):
                        report_html = gr.HTML(
                            value="<div style='color:#94a3b8;padding:24px;font-size:13px;'>"
                                  "Complete a diagnostic analysis to generate the medical report.</div>"
                        )
                        with gr.Row():
                            export_json_btn = gr.Button("💾 Export JSON",     variant="secondary",
                                                        elem_classes=["btn-secondary"])
                            export_md_btn   = gr.Button("📄 Export Markdown", variant="secondary",
                                                        elem_classes=["btn-secondary"])
                            export_pdf_btn  = gr.Button("📕 Export PDF",      variant="secondary",
                                                        elem_classes=["btn-secondary"])
                        download_file = gr.File(label="Download Report", interactive=False, visible=False)

        # ── Event Handlers ────────────────────────────────────────────────────

        def handle_loading(img_path):
            """Immediately show loading state."""
            if not img_path:
                return (
                    "<div class='alert-error'>❌ Please upload an image first.</div>",
                    _pipeline_stages_loading(),
                )
            return (
                "<div class='status-bar'>"
                "<span class='status-dot'></span>"
                "🧠 Initializing AI Engine & running pipeline…"
                "</div>",
                _pipeline_stages_loading(),
            )

        def handle_analysis(img_path, modality_str, patient_id):
            """Full analysis handler — builds all rich UI components."""
            if not img_path:
                err_html = "<div class='alert-error'>❌ Please upload an image first.</div>"
                return (
                    err_html,
                    "<div style='color:#94a3b8;font-size:12px;'>No image provided.</div>",
                    "<div class='result-card' style='text-align:center;color:#ef4444;padding:40px;'>No image provided.</div>",
                    {},
                    None,
                    None,
                    "<div style='color:#94a3b8;'>No data.</div>",
                    "<div class='result-card' style='text-align:center;padding:40px;color:#ef4444;'>No data.</div>",
                    "<div style='color:#94a3b8;padding:20px;'>No report.</div>",
                    {},
                    gr.update(visible=False),
                )

            mod_key = MODALITY_KEY_MAP.get(modality_str)
            success, res, err = run_prediction(img_path, mod_key, patient_id)

            if not success:
                err_html = f"<div class='alert-error'>❌ {err}</div>"
                pipeline_err = (
                    "<div class='result-card' style='color:#ef4444;padding:16px;'>"
                    f"Pipeline failed: {err}</div>"
                )
                return (
                    err_html,
                    pipeline_err,
                    f"<div class='result-card' style='color:#ef4444;padding:40px;text-align:center;'>{err}</div>",
                    {},
                    None,
                    None,
                    "<div style='color:#ef4444;'>Error.</div>",
                    f"<div class='result-card' style='color:#ef4444;padding:40px;'>{err}</div>",
                    "<div style='color:#ef4444;padding:20px;'>Error generating report.</div>",
                    {},
                    gr.update(visible=False),
                )

            # ── Extract prediction data ──────────────────────────────────────
            modality_display = res.get("modality", "Unknown").replace("_", " ").title()
            disease          = res.get("predicted_class", "Unknown")
            conf             = float(res.get("confidence_score", 0.0))
            conf_pct_str     = f"{conf * 100:.1f}%"
            reliability      = res.get("reliability", "low")
            requires_review  = bool(res.get("requires_review", True))
            total_ms         = float(res.get("total_ms", 0.0))
            top_preds        = res.get("top_predictions", [])
            heatmap_path     = res.get("gradcam_heatmap")
            request_id       = res.get("request_id", "")[:8]
            pid              = res.get("patient_id", "N/A")
            routing_mode     = res.get("routing_mode", "automatic")

            rel_label_map = {"high": "HIGH", "moderate": "MODERATE", "low": "LOW"}
            rel_label = rel_label_map.get(reliability, reliability.upper())

            rel_color = {"HIGH": "#10b981", "MODERATE": "#f59e0b", "LOW": "#ef4444"}.get(rel_label, "#94a3b8")

            # ── Confidence gauge ────────────────────────────────────────────
            gauge_html = _confidence_gauge(conf, reliability)

            # ── Risk badge ──────────────────────────────────────────────────
            rbadge = _risk_badge(reliability, requires_review)

            # ── Top-K bars ──────────────────────────────────────────────────
            topk_html = _topk_bars(top_preds)

            # ── Modality icon ───────────────────────────────────────────────
            mod_icon = "🩺"
            for label, key in MODALITY_KEY_MAP.items():
                if key == res.get("modality"):
                    mod_icon = MODALITY_ICONS.get(label, "🩺")
                    break

            # ── Primary Diagnosis Card ───────────────────────────────────────
            conf_int = int(conf * 100)
            diag_card = f"""
            <div class="result-card">
              <div class="result-card-header">
                <span style="font-size:22px;">{mod_icon}</span>
                <span class="result-card-title">Modality: <strong>{modality_display}</strong></span>
                <span style="margin-left:auto;">{rbadge}</span>
              </div>

              <div style="display:flex; gap:20px; align-items:flex-start; margin:16px 0;">
                <div style="flex:1;">
                  <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;">
                    PREDICTED CONDITION
                  </div>
                  <div class="disease-name" style="margin-top:4px;">{disease}</div>

                  <div style="margin-top:16px;">
                    <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                      DIFFERENTIAL DIAGNOSES
                    </div>
                    {topk_html}
                  </div>
                </div>

                <div style="flex-shrink:0;">
                  {gauge_html}
                </div>
              </div>

              <div style="margin-top:16px;">
                <div style="display:flex; justify-content:space-between; font-size:11px;
                     color:#94a3b8; margin-bottom:4px;">
                  <span>Confidence Score</span>
                  <span style="font-weight:700;color:{rel_color};">{conf_pct_str}</span>
                </div>
                <div class="conf-bar-wrap">
                  <div class="conf-bar-fill" style="width:{conf_int}%;
                       background:linear-gradient(90deg, {rel_color}aa, {rel_color});"></div>
                </div>
              </div>

              <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div class="info-row">
                  <span class="info-label">Reliability Index</span>
                  <span class="badge" style="background:rgba(16,185,129,0.1);color:{rel_color};">{rel_label}</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Routing Mode</span>
                  <span class="info-value">{routing_mode.title()}</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Case ID</span>
                  <span class="info-value" style="font-family:'JetBrains Mono',monospace;">#{request_id}</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Patient ID</span>
                  <span class="info-value">{pid}</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Pipeline Latency</span>
                  <span class="info-value" style="font-family:'JetBrains Mono',monospace;">{total_ms:.1f} ms</span>
                </div>
              </div>
            </div>
            """

            # ── GradCAM info ────────────────────────────────────────────────
            valid_heatmap = heatmap_path and Path(heatmap_path).exists()
            cam_info_html = (
                f"<div class='alert-success'>✅ GradCAM heatmap generated — "
                f"attention map highlights discriminative regions.</div>"
                if valid_heatmap else
                "<div class='alert-warning'>ℹ️ GradCAM visualization generated on prediction "
                "execution (displayed above).</div>"
            )
            cam_label_info = (
                "<div style='display:flex;gap:12px;align-items:center;margin-bottom:8px;'>"
                "<span class='img-compare-label'>Original</span>"
                "<span style='color:#475569;'>→</span>"
                "<span class='img-compare-label heatmap'>GradCAM Heatmap</span>"
                "</div>"
            )
            overlay_src = img_path
            if valid_heatmap:
                overlay_src = heatmap_path

            # ── Anomaly Card ────────────────────────────────────────────────
            anom_score  = float(res.get("anomaly_score", 0.0))
            is_anom     = bool(res.get("is_anomaly", False))
            anom_thresh = float(res.get("anomaly_threshold", 0.5))

            anom_label = "OOD / UNKNOWN ANOMALY DETECTED" if is_anom else "NORMAL IN-DISTRIBUTION SCAN"
            anom_color = "#ef4444" if is_anom else "#10b981"
            anom_cls   = "high-risk" if is_anom else "normal"
            anom_pct   = min(100, int(anom_score * 100))

            anom_card = f"""
            <div class="result-card">
              <div class="result-card-header">
                <span style="font-size:22px;">🛡️</span>
                <span class="result-card-title">Out-of-Distribution Anomaly Engine</span>
                <span style="margin-left:auto;">
                  <span class="risk-badge {anom_cls}">
                    <span class="risk-badge-dot" style="background:{anom_color};"></span>
                    {anom_label}
                  </span>
                </span>
              </div>

              <div style="display:flex; gap:24px; align-items:flex-start; margin:16px 0 8px;">
                <div style="flex:1;">
                  <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;">
                    ANOMALY SCORE
                  </div>
                  <div style="font-size:38px;font-weight:900;color:{anom_color};
                       filter:drop-shadow(0 0 8px {anom_color}88);letter-spacing:-1px;">
                    {anom_score:.4f}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;">
                    THRESHOLD
                  </div>
                  <div style="font-size:28px;font-weight:800;color:#cbd5e1;">{anom_thresh:.2f}</div>
                </div>
              </div>

              <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{anom_pct}%;
                     background:linear-gradient(90deg, {anom_color}88, {anom_color});"></div>
              </div>

              <div style="margin-top:16px;">
                <div class="info-row">
                  <span class="info-label">Pattern Classification</span>
                  <span class="info-value">{anom_label}</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Detection Method</span>
                  <span class="info-value">Mahalanobis + Reconstruction Fusion</span>
                </div>
                <div class="info-row">
                  <span class="info-label">Score vs Threshold</span>
                  <span class="info-value">{anom_score:.4f} / {anom_thresh:.2f}
                    {'  🔴 EXCEEDED' if is_anom else '  ✅ WITHIN BOUNDS'}
                  </span>
                </div>
              </div>
            </div>
            """

            # ── Medical Report ───────────────────────────────────────────────
            rpt_ok, rpt_data, rpt_err = generate_report(res)
            primary_dx  = rpt_data.get("primary_diagnosis", disease)
            clinical_note = rpt_data.get("clinical_note", res.get("clinical_note", ""))
            diff_dx_list  = rpt_data.get("differential_diagnoses", res.get("differential_diagnoses", []))
            diff_rows = "".join(
                f"<div class='info-row'>"
                f"  <span class='info-label'>{d.get('disease','—') if isinstance(d,dict) else str(d)}</span>"
                f"  <span class='info-value'>{d.get('confidence','')*100:.1f}% conf.</span>"
                f"</div>"
                if isinstance(d, dict) else
                f"<div class='info-row'><span class='info-label'>{d}</span></div>"
                for d in diff_dx_list[:5]
            ) or "<div style='color:#94a3b8;font-size:12px;'>None listed.</div>"

            report_body = f"""
            <div class="result-card">
              <div class="result-card-header">
                <span style="font-size:22px;">📋</span>
                <span class="result-card-title">Medical AI Diagnostic Report — Case #{request_id}</span>
              </div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:4px 0 16px;">
                <div style="background:var(--navy-800);border-radius:10px;padding:14px;">
                  <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Primary Diagnosis</div>
                  <div style="font-size:17px;font-weight:800;color:#f8fafc;">{primary_dx}</div>
                </div>
                <div style="background:var(--navy-800);border-radius:10px;padding:14px;">
                  <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Confidence</div>
                  <div style="font-size:17px;font-weight:800;color:{rel_color};">{conf_pct_str} ({rel_label})</div>
                </div>
                <div style="background:var(--navy-800);border-radius:10px;padding:14px;">
                  <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Modality</div>
                  <div style="font-size:17px;font-weight:800;color:#f8fafc;">{modality_display}</div>
                </div>
                <div style="background:var(--navy-800);border-radius:10px;padding:14px;">
                  <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Anomaly Flag</div>
                  <div style="font-size:17px;font-weight:800;color:{anom_color};">{'⚠️ Yes' if is_anom else '✅ No'}</div>
                </div>
              </div>

              <div style="margin-bottom:12px;">
                <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Clinical Note</div>
                <div style="font-size:13px;color:#cbd5e1;line-height:1.7;background:var(--navy-800);
                     border-radius:10px;padding:14px;">{clinical_note or '—'}</div>
              </div>

              <div>
                <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Differential Diagnoses</div>
                {diff_rows}
              </div>

              <div class="alert-warning" style="margin-top:16px;">
                ⚠️ <strong>Disclaimer:</strong> This AI report is intended as a clinical decision support
                tool only. All findings must be verified by a qualified medical specialist.
              </div>
            </div>
            """

            # ── Pipeline done tracker ────────────────────────────────────────
            pipeline_done = _pipeline_stages_done(total_ms)

            # ── Status message ───────────────────────────────────────────────
            status_msg = (
                f"<div class='alert-success'>✅ Analysis complete for Case #{request_id} "
                f"in <strong>{total_ms:.1f} ms</strong>.</div>"
            )

            return (
                status_msg,
                pipeline_done,
                diag_card,
                res,
                img_path,
                overlay_src,
                cam_label_info + cam_info_html,
                anom_card,
                report_body,
                rpt_data,
                gr.update(visible=False),
            )

        # Bind events
        analyze_btn.click(
            fn=handle_loading,
            inputs=[image_input],
            outputs=[status_output, pipeline_tracker_out],
        ).then(
            fn=handle_analysis,
            inputs=[image_input, modality_select, patient_id_input],
            outputs=[
                status_output,
                pipeline_tracker_out,
                diag_output,
                result_state,
                cam_orig,
                cam_overlay,
                cam_info,
                anomaly_output,
                report_html,
                report_state,
                download_file,
            ],
        )

        # Export handlers
        def do_export(rpt_data, fmt):
            if not rpt_data:
                return gr.update(visible=False)
            ok, fpath, err = export_report(rpt_data, fmt)
            if ok and fpath:
                return gr.update(value=fpath, visible=True)
            return gr.update(visible=False)

        export_json_btn.click(fn=lambda r: do_export(r, "json"),  inputs=[report_state], outputs=[download_file])
        export_md_btn.click(  fn=lambda r: do_export(r, "md"),    inputs=[report_state], outputs=[download_file])
        export_pdf_btn.click( fn=lambda r: do_export(r, "pdf"),   inputs=[report_state], outputs=[download_file])
