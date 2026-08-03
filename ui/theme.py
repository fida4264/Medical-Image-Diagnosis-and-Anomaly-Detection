"""
theme.py — Module 20: Professional Gradio Web Application
Medical AI Platform — Dark Navy Medical Dashboard Theme & Custom CSS.

Completely redesigned with:
  • Stunning landing hero section
  • Animated staged processing indicator
  • SVG circular confidence gauge with animated needle
  • Professional clinical risk badges with pulse
  • Image comparison slider (original / heatmap / overlay)
  • Demo mode overlay
  • Premium glassmorphism cards
  • Full-screen footer with credits
"""

from __future__ import annotations

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

MEDICAL_CSS = """
/* ═══════════════════════════════════════════════════════════════════
   0. GOOGLE FONTS
═══════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ═══════════════════════════════════════════════════════════════════
   1. DESIGN TOKENS
═══════════════════════════════════════════════════════════════════ */
:root {
    --navy-950: #020617;
    --navy-900: #0a0f1e;
    --navy-800: #0f172a;
    --navy-700: #1e293b;
    --navy-600: #334155;
    --med-blue: #0ea5e9;
    --med-blue-dark: #0284c7;
    --med-blue-light: #38bdf8;
    --cyan-400: #22d3ee;
    --cyan-300: #67e8f9;
    --emerald: #10b981;
    --emerald-dark: #059669;
    --amber: #f59e0b;
    --red-500: #ef4444;
    --red-400: #f87171;
    --purple: #7c3aed;
    --purple-light: #a78bfa;
    --white: #f8fafc;
    --text-muted: #94a3b8;
    --text-secondary: #cbd5e1;
    --border-color: #1e3a5f;
    --card-bg: #0d1b2e;
    --card-bg-2: rgba(13,27,46,0.95);
    --card-border: #1a3a5c;
    --hover-bg: #0f2540;
    --glass-bg: rgba(14,165,233,0.04);
    --gradient-1: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%);
    --gradient-2: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
    --gradient-3: linear-gradient(135deg, #7c3aed 0%, #0ea5e9 100%);
    --gradient-hero: linear-gradient(160deg, #020617 0%, #0a0f1e 50%, #0d1b2e 100%);
    --shadow: 0 4px 24px rgba(0,0,0,0.5);
    --shadow-card: 0 2px 16px rgba(14,165,233,0.08);
    --shadow-glow: 0 0 40px rgba(14,165,233,0.15);
    --radius: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    --transition-slow: all 0.5s cubic-bezier(0.4,0,0.2,1);
}

/* ═══════════════════════════════════════════════════════════════════
   2. GLOBAL RESET & BASE
═══════════════════════════════════════════════════════════════════ */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; box-sizing: border-box; }

body, .gradio-container, .main {
    background: var(--navy-900) !important;
    color: var(--white) !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* ═══════════════════════════════════════════════════════════════════
   3. HEADER BANNER
═══════════════════════════════════════════════════════════════════ */
.med-header {
    background: linear-gradient(90deg, #020617 0%, #0a0f1e 50%, #0d1b2e 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 16px 32px;
    display: flex; align-items: center; gap: 16px;
    position: sticky; top: 0; z-index: 200;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.5);
}
.med-header-logo {
    width: 46px; height: 46px;
    background: var(--gradient-1);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
    box-shadow: 0 0 24px rgba(14,165,233,0.4);
    animation: logo-pulse 3s ease-in-out infinite;
}
@keyframes logo-pulse {
    0%, 100% { box-shadow: 0 0 24px rgba(14,165,233,0.4); }
    50% { box-shadow: 0 0 40px rgba(14,165,233,0.6); }
}
.med-header-text h1 {
    font-size: 17px; font-weight: 800; color: var(--white);
    margin: 0; letter-spacing: -0.3px;
}
.med-header-text p { font-size: 11px; color: var(--text-muted); margin: 2px 0 0 0; }
.med-header-badge {
    margin-left: auto;
    background: rgba(14,165,233,0.12);
    border: 1px solid rgba(14,165,233,0.3);
    color: var(--med-blue-light);
    padding: 4px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
}
.header-status-dot {
    width: 8px; height: 8px; background: var(--emerald);
    border-radius: 50%; animation: pulse-dot 2s infinite; flex-shrink: 0;
}

/* ═══════════════════════════════════════════════════════════════════
   4. LANDING HERO PAGE
═══════════════════════════════════════════════════════════════════ */
.landing-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: var(--navy-950);
    z-index: 999; display: flex; align-items: center; justify-content: center;
    transition: opacity 0.6s ease, visibility 0.6s ease;
}
.landing-overlay.hidden { opacity: 0; visibility: hidden; pointer-events: none; }

.hero-container {
    background: var(--gradient-hero);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    padding: 60px 48px;
    text-align: center;
    max-width: 760px;
    width: 90%;
    position: relative; overflow: hidden;
    box-shadow: 0 40px 120px rgba(0,0,0,0.7), var(--shadow-glow);
}
.hero-container::before {
    content: ''; position: absolute;
    top: -60%; left: -20%; right: -20%; bottom: -20%;
    background: radial-gradient(ellipse at 50% 0%, rgba(14,165,233,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-grid-bg {
    position: absolute; inset: 0; pointer-events: none; opacity: 0.04;
    background-image: linear-gradient(rgba(14,165,233,0.5) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(14,165,233,0.5) 1px, transparent 1px);
    background-size: 40px 40px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.25);
    border-radius: 20px; padding: 6px 16px; margin-bottom: 28px;
    font-size: 12px; font-weight: 700; color: var(--med-blue-light);
    letter-spacing: 0.5px;
}
.hero-badge-dot { width: 7px; height: 7px; background: var(--emerald); border-radius: 50%; animation: pulse-dot 2s infinite; }
.hero-title {
    font-size: 44px; font-weight: 900; color: var(--white);
    letter-spacing: -2px; line-height: 1.1; margin: 0 0 12px 0;
    background: linear-gradient(135deg, #f8fafc 40%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-subtitle {
    font-size: 16px; color: var(--text-muted); margin: 0 0 40px 0; line-height: 1.6;
    max-width: 520px; margin-left: auto; margin-right: auto; margin-bottom: 40px;
}
.hero-pill-row {
    display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 40px;
}
.hero-pill {
    background: rgba(14,165,233,0.08); border: 1px solid rgba(14,165,233,0.2);
    border-radius: 20px; padding: 6px 14px; font-size: 12px;
    color: var(--text-secondary); font-weight: 600;
}
.hero-buttons { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }

.hero-btn-primary {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    border: none !important; border-radius: 12px !important;
    color: white !important; font-weight: 800 !important;
    font-size: 15px !important; padding: 14px 36px !important;
    cursor: pointer !important; transition: var(--transition) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.4) !important;
    letter-spacing: 0.3px !important;
}
.hero-btn-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 32px rgba(14,165,233,0.55) !important;
}
.hero-btn-demo {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: 12px !important; color: #34d399 !important;
    font-weight: 700 !important; font-size: 15px !important;
    padding: 14px 36px !important; cursor: pointer !important;
    transition: var(--transition) !important;
}
.hero-btn-demo:hover {
    background: rgba(16,185,129,0.2) !important;
    transform: translateY(-2px) !important;
}

.hero-stats-row {
    display: flex; gap: 32px; justify-content: center;
    margin-top: 40px; padding-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.hero-stat { text-align: center; }
.hero-stat-val { font-size: 24px; font-weight: 900; color: var(--white); letter-spacing: -1px; }
.hero-stat-lbl { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

/* ═══════════════════════════════════════════════════════════════════
   5. ANIMATED STAGED PROCESSING
═══════════════════════════════════════════════════════════════════ */
.pipeline-tracker {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin: 16px 0;
}
.pipeline-title {
    font-size: 11px; font-weight: 700; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;
}
.stage-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; font-size: 13px; color: var(--text-muted);
    opacity: 0.4; transition: var(--transition-slow);
}
.stage-row.active { opacity: 1; color: var(--med-blue-light); }
.stage-row.done { opacity: 1; color: var(--emerald); }
.stage-row.error { opacity: 1; color: var(--red-400); }
.stage-icon { font-size: 16px; width: 24px; text-align: center; flex-shrink: 0; }
.stage-spinner {
    width: 16px; height: 16px; flex-shrink: 0;
    border: 2px solid rgba(14,165,233,0.2);
    border-top-color: var(--med-blue);
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
.stage-check { color: var(--emerald); font-weight: 700; }
.stage-label { flex: 1; font-weight: 500; }
.stage-timing { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace !important; }

@keyframes spin { to { transform: rotate(360deg); } }

/* ═══════════════════════════════════════════════════════════════════
   6. CIRCULAR CONFIDENCE GAUGE (SVG-based)
═══════════════════════════════════════════════════════════════════ */
.gauge-container {
    display: flex; flex-direction: column; align-items: center;
    padding: 20px; gap: 12px;
}
.gauge-svg { width: 160px; height: 100px; overflow: visible; }
.gauge-label { font-size: 12px; color: var(--text-muted); font-weight: 600; text-align: center; }
.gauge-value { font-size: 28px; font-weight: 900; text-align: center; line-height: 1; }
.gauge-value.high { color: var(--emerald); }
.gauge-value.moderate { color: var(--amber); }
.gauge-value.low { color: var(--red-400); }

/* ═══════════════════════════════════════════════════════════════════
   7. CLINICAL RISK BADGES (ENHANCED)
═══════════════════════════════════════════════════════════════════ */
.risk-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 18px; border-radius: 24px;
    font-size: 13px; font-weight: 800; letter-spacing: 0.5px;
    border: 1px solid; position: relative; overflow: hidden;
}
.risk-badge::after {
    content: ''; position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: badge-shine 3s ease infinite;
}
@keyframes badge-shine {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.risk-badge-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse-dot 1.5s infinite; }
.risk-badge.normal {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.35);
    color: #34d399;
}
.risk-badge.normal .risk-badge-dot { background: var(--emerald); }
.risk-badge.warning {
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.35);
    color: #fbbf24;
}
.risk-badge.warning .risk-badge-dot { background: var(--amber); }
.risk-badge.high-risk {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.35);
    color: var(--red-400);
}
.risk-badge.high-risk .risk-badge-dot { background: var(--red-500); }
.risk-badge.critical {
    background: rgba(124,58,237,0.15);
    border-color: rgba(124,58,237,0.4);
    color: var(--purple-light);
}
.risk-badge.critical .risk-badge-dot { background: var(--purple); }

/* ═══════════════════════════════════════════════════════════════════
   8. IMAGE COMPARISON VIEWER
═══════════════════════════════════════════════════════════════════ */
.img-compare-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; margin: 8px 0;
}
.img-compare-card {
    background: var(--navy-800);
    border: 1px solid var(--card-border);
    border-radius: var(--radius); overflow: hidden;
    transition: var(--transition);
}
.img-compare-card:hover {
    border-color: var(--med-blue);
    box-shadow: 0 0 20px rgba(14,165,233,0.1);
}
.img-compare-label {
    padding: 8px 12px;
    font-size: 11px; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;
    background: var(--navy-800);
    border-bottom: 1px solid var(--card-border);
}
.img-compare-label.heatmap { color: #fb923c; }
.img-compare-label.overlay { color: var(--med-blue-light); }

/* ═══════════════════════════════════════════════════════════════════
   9. DEMO MODE OVERLAY
═══════════════════════════════════════════════════════════════════ */
.demo-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(14,165,233,0.1));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: var(--radius); padding: 16px 20px;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 12px;
}
.demo-banner-icon { font-size: 20px; }
.demo-banner-text { flex: 1; }
.demo-banner-text strong { color: #34d399; font-weight: 700; display: block; font-size: 13px; }
.demo-banner-text span { color: var(--text-muted); font-size: 12px; }

.demo-selector {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg); padding: 24px;
    margin: 12px 0;
}
.demo-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px; margin-top: 12px;
}
.demo-modality-btn {
    background: var(--navy-800);
    border: 1px solid var(--card-border);
    border-radius: 10px; padding: 14px 8px;
    text-align: center; cursor: pointer;
    transition: var(--transition); user-select: none;
}
.demo-modality-btn:hover, .demo-modality-btn.selected {
    background: rgba(14,165,233,0.12);
    border-color: var(--med-blue);
}
.demo-modality-btn .dmi { font-size: 28px; display: block; margin-bottom: 6px; }
.demo-modality-btn .dml { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

/* ═══════════════════════════════════════════════════════════════════
   10. SECTION & TAB STYLES
═══════════════════════════════════════════════════════════════════ */
.section-title {
    font-size: 22px; font-weight: 800; color: var(--white);
    margin: 0 0 4px 0; letter-spacing: -0.5px;
}
.section-sub { font-size: 13px; color: var(--text-muted); margin: 0; }

.tabs.svelte-1occ011 { background: transparent !important; }
button.svelte-1occ011 {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: var(--transition) !important;
    padding: 10px 18px !important;
}
button.svelte-1occ011.selected, button.svelte-1occ011:hover {
    background: rgba(14,165,233,0.15) !important;
    border-color: var(--med-blue) !important;
    color: var(--med-blue-light) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   11. STAT CARDS
═══════════════════════════════════════════════════════════════════ */
.stat-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important; transition: var(--transition) !important;
    position: relative; overflow: hidden;
}
.stat-card:hover {
    border-color: var(--med-blue) !important;
    box-shadow: 0 0 28px rgba(14,165,233,0.12) !important;
    transform: translateY(-2px);
}
.stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--gradient-1);
}
.stat-card-icon { font-size: 28px; margin-bottom: 12px; display: block; }
.stat-card-value { font-size: 32px; font-weight: 800; color: var(--white); letter-spacing: -1px; margin: 0; line-height: 1; }
.stat-card-label { font-size: 11px; color: var(--text-muted); margin: 6px 0 0 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

/* ═══════════════════════════════════════════════════════════════════
   12. UPLOAD ZONE
═══════════════════════════════════════════════════════════════════ */
.upload-zone {
    border: 2px dashed var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--navy-800) !important;
    min-height: 240px !important;
    transition: var(--transition) !important;
}
.upload-zone:hover {
    border-color: var(--med-blue) !important;
    background: rgba(14,165,233,0.04) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   13. ACTION BUTTONS
═══════════════════════════════════════════════════════════════════ */
.btn-primary {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px 32px !important;
    cursor: pointer !important; transition: var(--transition) !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(14,165,233,0.3) !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,0.45) !important;
}
.btn-secondary {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: var(--transition) !important;
}
.btn-secondary:hover {
    border-color: var(--med-blue) !important;
    color: var(--med-blue-light) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   14. RESULT CARDS (ENHANCED)
═══════════════════════════════════════════════════════════════════ */
.result-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important; transition: var(--transition) !important;
    animation: card-appear 0.4s ease;
}
@keyframes card-appear {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px; padding-bottom: 12px;
    border-bottom: 1px solid var(--navy-700);
}
.result-card-title {
    font-size: 12px; font-weight: 700; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 1px;
}
.disease-name {
    font-size: 26px; font-weight: 800; color: var(--white);
    letter-spacing: -0.5px;
}
.confidence-value {
    font-size: 38px; font-weight: 900;
    background: var(--gradient-1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* ═══════════════════════════════════════════════════════════════════
   15. CONFIDENCE BAR
═══════════════════════════════════════════════════════════════════ */
.conf-bar-wrap {
    background: var(--navy-700); border-radius: 6px;
    height: 10px; overflow: hidden; margin: 10px 0;
}
.conf-bar-fill {
    height: 100%; border-radius: 6px;
    background: var(--gradient-1);
    transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
}

/* TOP-K Prediction Bars */
.topk-row { margin: 6px 0; }
.topk-label-row {
    display: flex; justify-content: space-between;
    font-size: 12px; margin-bottom: 4px;
}
.topk-label { color: var(--text-secondary); font-weight: 500; }
.topk-pct { color: var(--text-muted); font-family: 'JetBrains Mono', monospace !important; }
.topk-bar-wrap { background: var(--navy-700); border-radius: 4px; height: 6px; }
.topk-bar-fill { height: 100%; border-radius: 4px; background: var(--gradient-1); }

/* ═══════════════════════════════════════════════════════════════════
   16. BADGE SYSTEM
═══════════════════════════════════════════════════════════════════ */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.3px;
}
.badge-success { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-warning { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-danger { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-info { background: rgba(14,165,233,0.15); color: #38bdf8; border: 1px solid rgba(14,165,233,0.3); }
.badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }

/* ═══════════════════════════════════════════════════════════════════
   17. MODALITY GRID
═══════════════════════════════════════════════════════════════════ */
.modality-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 12px; margin: 12px 0;
}
.modality-pill {
    background: var(--navy-800); border: 1px solid var(--card-border);
    border-radius: 10px; padding: 14px 8px;
    text-align: center; cursor: pointer; transition: var(--transition);
}
.modality-pill:hover, .modality-pill.active {
    background: rgba(14,165,233,0.12);
    border-color: var(--med-blue); color: var(--med-blue-light);
    transform: translateY(-2px);
}
.modality-pill .mp-icon { font-size: 26px; display: block; margin-bottom: 6px; }
.modality-pill .mp-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

/* ═══════════════════════════════════════════════════════════════════
   18. DATA TABLE
═══════════════════════════════════════════════════════════════════ */
table { width: 100% !important; border-collapse: collapse !important; }
th {
    background: var(--navy-800) !important;
    color: var(--text-muted) !important;
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    padding: 12px 16px !important; text-align: left !important;
    border-bottom: 1px solid var(--border-color) !important;
}
td {
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(30,58,95,0.5) !important;
    font-size: 13px !important; color: var(--text-secondary) !important;
    vertical-align: middle !important;
}
tr:hover td { background: var(--hover-bg) !important; }

/* ═══════════════════════════════════════════════════════════════════
   19. CHART / GAUGE CONTAINERS
═══════════════════════════════════════════════════════════════════ */
.chart-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
}
.chart-title {
    font-size: 12px; font-weight: 700; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin: 0 0 16px 0;
}

/* ═══════════════════════════════════════════════════════════════════
   20. ALERT MESSAGES
═══════════════════════════════════════════════════════════════════ */
.alert-warning {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important; font-size: 12px !important;
    color: #fbbf24 !important; line-height: 1.6 !important;
}
.alert-error {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important; font-size: 12px !important;
    color: #f87171 !important;
}
.alert-success {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important; font-size: 12px !important;
    color: #34d399 !important;
}

/* ═══════════════════════════════════════════════════════════════════
   21. INFO ROWS
═══════════════════════════════════════════════════════════════════ */
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid rgba(30,58,95,0.5);
}
.info-row:last-child { border-bottom: none; }
.info-label { font-size: 12px; color: var(--text-muted); font-weight: 500; }
.info-value { font-size: 13px; color: var(--white); font-weight: 600; }

/* ═══════════════════════════════════════════════════════════════════
   22. ABOUT PAGE
═══════════════════════════════════════════════════════════════════ */
.about-hero {
    background: var(--gradient-hero);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg); padding: 40px; text-align: center;
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.about-hero-title {
    font-size: 28px; font-weight: 900; color: var(--white);
    letter-spacing: -0.5px; margin: 0 0 8px 0;
}
.about-hero-sub { font-size: 14px; color: var(--text-muted); margin: 0; }

/* ═══════════════════════════════════════════════════════════════════
   23. STATUS / ANIMATIONS
═══════════════════════════════════════════════════════════════════ */
.status-bar {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 8px; padding: 8px 16px;
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: var(--emerald);
}
.status-dot {
    width: 8px; height: 8px; background: var(--emerald);
    border-radius: 50%; animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ═══════════════════════════════════════════════════════════════════
   24. PROFESSIONAL FOOTER
═══════════════════════════════════════════════════════════════════ */
.med-footer {
    background: var(--navy-950);
    border-top: 1px solid var(--border-color);
    padding: 32px;
    margin-top: 48px;
}
.med-footer-inner {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 16px;
}
.med-footer-brand { font-size: 14px; font-weight: 700; color: var(--white); }
.med-footer-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.med-footer-links { display: flex; gap: 24px; }
.med-footer-links a {
    font-size: 12px; color: var(--text-muted); text-decoration: none;
    font-weight: 500; transition: var(--transition);
}
.med-footer-links a:hover { color: var(--med-blue-light); }
.med-footer-disclaimer {
    font-size: 11px; color: rgba(148,163,184,0.5);
    text-align: center; margin-top: 20px; padding-top: 20px;
    border-top: 1px solid rgba(30,58,95,0.3); line-height: 1.6;
}

/* ═══════════════════════════════════════════════════════════════════
   25. GRADIO COMPONENT OVERRIDES
═══════════════════════════════════════════════════════════════════ */
.gradio-container { background: var(--navy-900) !important; }
.dark { background: var(--navy-900) !important; }
.block { background: transparent !important; border: none !important; }
.gr-prose { color: var(--text-secondary) !important; }
.gr-text-input, .gr-number, .gr-dropdown {
    background: var(--navy-800) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--white) !important; border-radius: 8px !important;
}
.gr-dropdown option { background: var(--navy-800) !important; }
label.svelte-1b6s6s, label { color: var(--text-secondary) !important; font-weight: 500 !important; }
.prose h1, .prose h2, .prose h3 { color: var(--white) !important; }
.prose p { color: var(--text-secondary) !important; }
footer.svelte-1rjryqp { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════
   26. LANDING ANIMATION
═══════════════════════════════════════════════════════════════════ */
@keyframes float-in {
    from { opacity: 0; transform: scale(0.95) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.hero-container { animation: float-in 0.7s ease forwards; }

/* ═══════════════════════════════════════════════════════════════════
   27. BENCHMARK TABLE STYLES
═══════════════════════════════════════════════════════════════════ */
.bm-table-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg); overflow: hidden;
}
.bm-model-row {
    display: flex; align-items: center;
    padding: 14px 20px; gap: 16px;
    border-bottom: 1px solid rgba(30,58,95,0.5);
    transition: var(--transition);
}
.bm-model-row:last-child { border-bottom: none; }
.bm-model-row:hover { background: var(--hover-bg); }
.bm-rank { font-size: 18px; font-weight: 900; color: var(--text-muted); width: 28px; text-align: center; }
.bm-rank.first { color: #fbbf24; }
.bm-rank.second { color: #94a3b8; }
.bm-rank.third { color: #cd7c3d; }
.bm-modality { font-size: 13px; font-weight: 700; color: var(--white); flex: 1; }
.bm-metric { text-align: right; }
.bm-metric-val { font-size: 16px; font-weight: 800; color: var(--white); }
.bm-metric-lbl { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
.bm-bar-wrap { width: 80px; background: var(--navy-700); border-radius: 4px; height: 6px; }
.bm-bar-fill { height: 100%; border-radius: 4px; background: var(--gradient-1); }
"""


def get_theme():
    """Build Gradio theme with medical dark navy styling."""
    if not GRADIO_AVAILABLE:
        return None
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )


__all__ = ["MEDICAL_CSS", "get_theme"]
