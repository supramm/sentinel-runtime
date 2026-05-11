"""
SentinelAI — Isolation Forest Forensic Engine
Headless Streamlit page, iframe-embedded in React.
"""

import streamlit as st
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI — Forensic Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.path_manager import PathManager

# ─────────────────────────────────────────────
# CENTRALIZED PATH MANAGEMENT
# ─────────────────────────────────────────────
pm = PathManager()
pm.print_paths()  # Debug output at startup

# ─────────────────────────────────────────────
# FEATURE CONFIG — unchanged
# ─────────────────────────────────────────────
IF_FEATURE_NAMES = [
    "temperature_c", "vibration_mm_s", "pressure_kpa", "motor_rpm",
    "flow_rate_lpm", "power_consumption_kw", "coolant_temp_c",
    "acoustic_level_db", "oil_viscosity_cst",
    "cooling_stress", "power_load_proxy", "pressure_to_flow_ratio", "mode_enc",
]

SUBSYSTEMS = {
    "THERMAL":     ["temperature_c", "coolant_temp_c", "oil_viscosity_cst", "cooling_stress"],
    "MECHANICAL":  ["vibration_mm_s", "acoustic_level_db", "motor_rpm"],
    "HYDRAULIC":   ["pressure_kpa", "flow_rate_lpm", "pressure_to_flow_ratio"],
    "ELECTRICAL":  ["power_consumption_kw", "power_load_proxy"],
    "OPERATIONAL": ["mode_enc"],
}
SUBSYSTEM_COLORS = {
    "THERMAL":     "#ff643c",
    "MECHANICAL":  "#00ffc8",
    "HYDRAULIC":   "#64c864",
    "ELECTRICAL":  "#ffc800",
    "OPERATIONAL": "#b482ff",
}

# ─────────────────────────────────────────────
# CSS — cinematic glassmorphism (RUL-matched)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── hide Streamlit chrome ── */
#MainMenu, footer, header, .stToolbar,
[data-testid="stToolbar"], [data-testid="stHeader"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

/* ── transparent root ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background: transparent !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] { display: none !important; }

/* ── base dark cinematic bg ── */
body {
    background: radial-gradient(ellipse at 20% 30%,
        rgba(255,100,0,0.04) 0%,
        rgba(0,80,255,0.03) 40%,
        #030712 100%) !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    color: #e2e8f0 !important;
}

/* ── headings / text ── */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div {
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ── metric block ── */
.metric-block {
    background: rgba(255,140,0,0.04);
    border: 1px solid rgba(255,140,0,0.18);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: rgba(255,140,0,0.65) !important;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ff8c00 !important;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.6rem;
    color: rgba(255,255,255,0.35) !important;
    margin-top: 0.15rem;
}

/* ── generate button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(255,140,0,0.12), rgba(200,80,0,0.12)) !important;
    border: 1px solid rgba(255,140,0,0.35) !important;
    color: #ff8c00 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 20px rgba(255,140,0,0.1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(255,140,0,0.22), rgba(200,80,0,0.22)) !important;
    box-shadow: 0 0 35px rgba(255,140,0,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── divider ── */
hr { border-color: rgba(255,140,0,0.08) !important; }
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,0,0.15), transparent);
    margin: 0.8rem 0;
}

/* ── section label ── */
.section-label {
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: rgba(255,140,0,0.45) !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: rgba(255,140,0,0.4);
}

/* ── stat pills ── */
.stat-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}
.stat-pill {
    font-size: 0.58rem;
    padding: 0.2rem 0.7rem;
    background: rgba(255,140,0,0.06);
    border: 1px solid rgba(255,140,0,0.15);
    border-radius: 999px;
    color: rgba(255,140,0,0.7) !important;
    letter-spacing: 0.08em;
}

/* ── status hero ── */
.status-hero {
    text-align: center;
    padding: 1.4rem 0 0.6rem;
}
.status-icon {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    line-height: 1;
}
.status-main {
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 900;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-top: 0.3rem;
}
.status-sub-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.35rem;
    opacity: 0.6;
}
.state-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── error card ── */
.error-card {
    background: rgba(255,60,60,0.05);
    border: 1px solid rgba(255,60,60,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
}

/* ── plotly container ── */
.js-plotly-plot .plotly, .plot-container {
    background: transparent !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,140,0,0.2); border-radius: 4px; }

/* ── animations ── */
@keyframes pulse-forensic {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(255,59,92,0.6)); }
    50%       { filter: drop-shadow(0 0 28px rgba(255,59,92,0.95)); }
}
@keyframes pulse-marginal {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(255,140,0,0.5)); }
    50%       { filter: drop-shadow(0 0 24px rgba(255,140,0,0.85)); }
}
@keyframes header-pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY BASE LAYOUT — cinematic transparent
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, Courier New, monospace", color="#94a3b8", size=10),
    margin=dict(l=42, r=16, t=30, b=32),
    xaxis=dict(
        gridcolor="rgba(255,140,0,0.07)",
        zerolinecolor="rgba(255,140,0,0.12)",
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9),
    ),
    yaxis=dict(
        gridcolor="rgba(255,140,0,0.07)",
        zerolinecolor="rgba(255,140,0,0.12)",
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9),
    ),
)

STATE_COLORS = {
    "SYSTEM STABLE":            {"fg": "#00ffc8", "bg": "rgba(0,255,200,0.07)",
                                 "border": "rgba(0,255,200,0.25)", "glow": "rgba(0,255,200,0.4)"},
    "SENSOR DEVIATION DETECTED":{"fg": "#ffb400", "bg": "rgba(255,180,0,0.07)",
                                 "border": "rgba(255,180,0,0.3)",  "glow": "rgba(255,180,0,0.4)"},
    "ANOMALOUS PATTERN FOUND":  {"fg": "#ff3b5c", "bg": "rgba(255,59,92,0.07)",
                                 "border": "rgba(255,59,92,0.3)",  "glow": "rgba(255,59,92,0.45)"},
}


def _layout(**overrides):
    layout = dict(PLOTLY_BASE)
    layout.update(overrides)
    return layout


# ─────────────────────────────────────────────
# RESOURCE LOADING — unchanged
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_if_resources():
    import joblib
    try:
        model     = joblib.load(pm.get_pkl_path("if", "if_model.pkl"))
        scaler    = joblib.load(pm.get_pkl_path("if", "if_scaler.pkl"))
        threshold_raw = joblib.load(pm.get_pkl_path("if", "if_threshold.pkl"))
        threshold = float(threshold_raw) if not isinstance(threshold_raw, dict) \
                    else float(list(threshold_raw.values())[0])
        with open(pm.get_json_path("if", "if_feature_indices.json"), "r") as f:
            feat_indices = json.load(f)
        means = np.load(pm.get_npy_path("if", "if_feature_means.npy"))
        return model, scaler, threshold, feat_indices, means
    except Exception as e:
        st.error(f"Failed to load IF resources: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_data():
    try:
        return np.load(pm.get_data_path("X_test_all.npy")).astype(np.float32)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        raise


# ─────────────────────────────────────────────
# INFERENCE — unchanged
# ─────────────────────────────────────────────
def extract_if_features(window, feat_indices):
    last_step = window[-1, :]
    return last_step[feat_indices]


def run_if_inference(window, model, scaler, threshold, feat_indices, means):
    feats        = extract_if_features(window, feat_indices)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    score        = float(model.decision_function(feats_scaled)[0])
    prediction   = int(model.predict(feats_scaled)[0])
    is_anomaly   = (score < threshold)

    n_feat = len(feat_indices)
    if means.shape[0] == n_feat:
        raw_deviation = feats - means[:n_feat]
    else:
        raw_deviation = feats - means

    try:
        feat_std      = np.sqrt(scaler.var_)
        norm_deviation = raw_deviation / (feat_std + 1e-9)
    except AttributeError:
        norm_deviation = raw_deviation / (np.abs(means[:n_feat]) + 1e-9)

    contributions = np.abs(norm_deviation)
    top_idx       = np.argsort(contributions)[::-1][:3]
    top_features  = [(IF_FEATURE_NAMES[i], float(contributions[i]), float(norm_deviation[i]))
                     for i in top_idx]

    subsystem_scores = {}
    for sys_name, sys_feats in SUBSYSTEMS.items():
        total = 0.0
        for fname in sys_feats:
            if fname in IF_FEATURE_NAMES:
                fi = IF_FEATURE_NAMES.index(fname)
                if fi < len(contributions):
                    total += contributions[fi]
        subsystem_scores[sys_name] = total
    top_subsystem = max(subsystem_scores, key=subsystem_scores.get)

    if not is_anomaly and score > 0.05:
        status = "SYSTEM STABLE"
        status_icon = "◉"
    elif not is_anomaly:
        status = "SENSOR DEVIATION DETECTED"
        status_icon = "◈"
    else:
        status = "ANOMALOUS PATTERN FOUND"
        status_icon = "⬟"

    score_range      = 0.226 - (-0.176)
    normalized_score = (score - (-0.176)) / score_range
    if is_anomaly:
        confidence = (1 - normalized_score) * 100
    else:
        confidence = normalized_score * 100

    return {
        "score":            score,
        "threshold":        threshold,
        "is_anomaly":       is_anomaly,
        "status":           status,
        "status_icon":      status_icon,
        "confidence":       min(confidence, 99.9),
        "contributions":    contributions,
        "norm_deviation":   norm_deviation,
        "raw_deviation":    raw_deviation,
        "top_features":     top_features,
        "subsystem_scores": subsystem_scores,
        "top_subsystem":    top_subsystem,
        "feats_raw":        feats,
    }


# ─────────────────────────────────────────────
# CHARTS — cinematic versions
# ─────────────────────────────────────────────
def make_score_gauge(score, threshold):
    """Radial gauge for IF isolation score — matches RUL gauge style."""
    min_s, max_s = -0.20, 0.25
    # Normalise to 0–100 scale for the indicator
    range_s  = max_s - min_s
    pct      = (score - min_s) / range_s * 100   # 0–100
    thr_pct  = (threshold - min_s) / range_s * 100

    is_anomaly = score < threshold
    color = "#ff3b5c" if is_anomaly else "#00ffc8"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 4),
        number=dict(
            font=dict(size=30, color=color,
                      family="JetBrains Mono, Courier New, monospace"),
            valueformat="+.4f",
        ),
        gauge=dict(
            axis=dict(
                range=[min_s, max_s],
                tickfont=dict(color="rgba(255,255,255,0.3)", size=8),
                tickcolor="rgba(255,255,255,0.15)",
                tickformat="+.2f",
            ),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[min_s, threshold],  color="rgba(255,59,92,0.07)"),
                dict(range=[threshold, max_s],   color="rgba(0,255,200,0.05)"),
            ],
            threshold=dict(
                line=dict(color="rgba(255,200,0,0.7)", width=2),
                thickness=0.75,
                value=threshold,
            ),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    layout = dict(PLOTLY_BASE)
    layout["margin"] = dict(l=20, r=20, t=10, b=10)
    layout["height"]  = 220
    fig.update_layout(**layout)
    return fig


def make_deviation_bars(contributions, norm_deviation, n_feat):
    feat_labels = IF_FEATURE_NAMES[:n_feat]
    sorted_idx  = np.argsort(contributions)  # ascending → horizontal bar reads top=worst

    max_c = contributions.max() if contributions.max() > 0 else 1
    colors = []
    for ci in sorted_idx:
        r = contributions[ci] / max_c
        if r > 0.7:
            colors.append("#ff3b5c")
        elif r > 0.4:
            colors.append("#ff8c00")
        else:
            colors.append("rgba(0,255,200,0.6)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contributions[sorted_idx],
        y=[feat_labels[i] for i in sorted_idx],
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=240,
        showlegend=False,
        title=dict(
            text="FEATURE DEVIATION MAGNITUDE",
            font=dict(size=9, color="rgba(255,140,0,0.45)"), x=0.01,
        ),
        xaxis=dict(
            title=dict(text="Normalized Deviation", font=dict(size=8)),
            gridcolor="rgba(255,140,0,0.07)",
            zerolinecolor="rgba(255,140,0,0.12)",
            tickfont=dict(size=8),
        ),
        yaxis=dict(tickfont=dict(size=8), gridcolor="rgba(255,140,0,0.05)"),
        bargap=0.18,
    ))
    return fig


def make_subsystem_radar(subsystem_scores):
    labels = list(subsystem_scores.keys())
    values = [subsystem_scores[k] for k in labels]
    max_v  = max(values) if max(values) > 0 else 1
    norm   = [v / max_v for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm + [norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(255,100,60,0.07)",
        line=dict(color="#ff643c", width=1.6),
        mode="lines+markers",
        marker=dict(size=5, color="#ff643c"),
        name="Subsystem Load",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Courier New, monospace", color="#94a3b8", size=9),
        margin=dict(l=40, r=40, t=36, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(255,100,60,0.08)",
                linecolor="rgba(255,100,60,0.12)",
                tickfont=dict(size=7),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor="rgba(255,100,60,0.08)",
                linecolor="rgba(255,100,60,0.15)",
                tickfont=dict(size=8, color="#94a3b8"),
            ),
        ),
        title=dict(
            text="SUBSYSTEM LOAD RADAR",
            font=dict(size=9, color="rgba(255,140,0,0.45)"), x=0.01,
        ),
        showlegend=False,
        height=250,
    )
    return fig


def make_decision_space(contributions, score, threshold):
    n           = len(contributions)
    feat_labels = IF_FEATURE_NAMES[:n]
    max_c       = contributions.max() if contributions.max() > 0 else 1

    marker_colors = []
    for c in contributions:
        r = c / max_c
        if r > 0.7:
            marker_colors.append("#ff3b5c")
        elif r > 0.4:
            marker_colors.append("#ff8c00")
        else:
            marker_colors.append("rgba(0,255,200,0.65)")

    fig = go.Figure()
    # Zone shading
    fig.add_shape(type="rect", x0=-0.5, x1=n-0.5, y0=0, y1=max_c*0.35,
                  fillcolor="rgba(0,255,140,0.04)", line=dict(color="rgba(0,255,140,0.1)"))
    fig.add_shape(type="rect", x0=-0.5, x1=n-0.5, y0=max_c*0.35, y1=max_c*0.65,
                  fillcolor="rgba(255,180,0,0.04)", line=dict(color="rgba(0,0,0,0)"))
    fig.add_shape(type="rect", x0=-0.5, x1=n-0.5, y0=max_c*0.65, y1=max_c*1.15,
                  fillcolor="rgba(255,60,60,0.04)", line=dict(color="rgba(0,0,0,0)"))

    fig.add_trace(go.Scatter(
        x=list(range(n)),
        y=list(contributions),
        mode="markers+lines",
        marker=dict(
            color=marker_colors, size=8, symbol="circle",
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        ),
        line=dict(color="rgba(255,140,0,0.18)", width=1, dash="dot"),
        text=feat_labels,
        hovertemplate="<b>%{text}</b><br>Contribution: %{y:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=max_c * 0.65,
        line=dict(color="rgba(255,59,92,0.3)", dash="dash", width=1),
    )
    fig.update_layout(**_layout(
        height=210,
        showlegend=False,
        title=dict(
            text=f"DECISION SPACE  ·  IF SCORE {score:+.4f}",
            font=dict(size=9, color="rgba(255,140,0,0.45)"), x=0.01,
        ),
        xaxis=dict(
            tickvals=list(range(n)),
            ticktext=[f.replace("_", " ").upper()[:8] for f in feat_labels],
            tickangle=-45, tickfont=dict(size=7),
            gridcolor="rgba(255,140,0,0.06)",
        ),
        yaxis=dict(
            title=dict(text="Contribution", font=dict(size=8)),
            gridcolor="rgba(255,140,0,0.06)", tickfont=dict(size=8),
        ),
    ))
    return fig


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "if_result" not in st.session_state:
    st.session_state.if_result = None
if "if_rng" not in st.session_state:
    st.session_state.if_rng = np.random.default_rng()

# ─────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.4rem 0.2rem 0.8rem;
            border-bottom:1px solid rgba(255,140,0,0.08);
            margin-bottom:0.8rem;">
  <div style="display:flex;align-items:center;gap:0.7rem;">
    <div style="width:8px;height:8px;border-radius:50%;background:#ff8c00;
                box-shadow:0 0 10px #ff8c00;
                animation:header-pulse 1.8s ease-in-out infinite;"></div>
    <span style="font-size:0.65rem;letter-spacing:0.25em;
                 color:rgba(255,140,0,0.75);text-transform:uppercase;">
      SentinelAI · Forensic Engine
    </span>
  </div>
  <div style="display:flex;gap:1.2rem;">
    <span style="font-size:0.58rem;color:rgba(255,255,255,0.25);">ISOLATION FOREST</span>
    <span style="font-size:0.58rem;color:rgba(255,255,255,0.25);">ROOT CAUSE ATTRIBUTION</span>
    <span style="font-size:0.58rem;color:rgba(255,140,0,0.45);">TATA TECH · JAMSHEDPUR</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
load_error = None
if_model = data = scaler = threshold = feat_indices = means = None

try:
    if_model, scaler, threshold, feat_indices, means = load_if_resources()
    data = load_data()
except Exception as e:
    load_error = str(e)

# ─────────────────────────────────────────────
# STAT PILLS + BUTTON ROW
# ─────────────────────────────────────────────
if not load_error and data is not None:
    n_if = len(feat_indices) if isinstance(feat_indices, list) \
           else len(feat_indices.get("indices", []))
    st.markdown(f"""
    <div class="stat-row">
      <span class="stat-pill">13 IF FEATURES</span>
      <span class="stat-pill">5 SUBSYSTEMS</span>
      <span class="stat-pill">{data.shape[0]:,} WINDOWS</span>
      <span class="stat-pill">THRESHOLD {threshold:+.4f}</span>
      <span class="stat-pill">WINDOW 50 CYCLES</span>
      <span class="stat-pill">LIVE INFERENCE</span>
    </div>
    """, unsafe_allow_html=True)

gen_col, info_col = st.columns([1, 3])
with gen_col:
    run = st.button("⟳  INVESTIGATE SCENARIO", key="if_gen_btn", use_container_width=True)
with info_col:
    if load_error:
        st.markdown(
            f'<div class="error-card">'
            f'<span class="metric-label">⚠ LOAD ERROR</span><br>'
            f'<span style="font-size:0.65rem;color:#ff8c8c;">{load_error}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# RUN INFERENCE — unchanged logic
# ─────────────────────────────────────────────
if run and if_model is not None and data is not None:
    idx    = st.session_state.if_rng.integers(0, data.shape[0])
    window = data[idx]

    fi = feat_indices if isinstance(feat_indices, list) \
         else feat_indices.get("indices", list(feat_indices.values())[0])

    result = run_if_inference(window, if_model, scaler, threshold, fi, means)
    result["window_idx"] = int(idx)
    result["n_feat"]     = len(fi)
    st.session_state.if_result = result

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# OUTPUT DISPLAY
# ─────────────────────────────────────────────
res = st.session_state.if_result

if res is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;opacity:0.4;">
      <div style="font-size:3rem;margin-bottom:1rem;
                  filter:drop-shadow(0 0 20px rgba(255,140,0,0.3));">◈</div>
      <div style="font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;
                  color:rgba(255,140,0,0.6);">Forensic Engine Standby</div>
      <div style="font-size:0.55rem;color:rgba(255,255,255,0.25);
                  margin-top:0.5rem;letter-spacing:0.1em;">
        Press INVESTIGATE SCENARIO to begin root-cause analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    state = res["status"]
    sc    = STATE_COLORS.get(state, STATE_COLORS["SYSTEM STABLE"])

    pulse_map = {
        "SYSTEM STABLE":             "",
        "SENSOR DEVIATION DETECTED": "animation:pulse-marginal 2s ease-in-out infinite;",
        "ANOMALOUS PATTERN FOUND":   "animation:pulse-forensic 1.8s ease-in-out infinite;",
    }
    sub_map = {
        "SYSTEM STABLE":             f"ALL SUBSYSTEMS NOMINAL · SUBSYSTEM: {res['top_subsystem']}",
        "SENSOR DEVIATION DETECTED": f"DEVIATION LOGGED · PRIMARY: {res['top_subsystem']}",
        "ANOMALOUS PATTERN FOUND":   f"ANOMALY CONFIRMED · ROOT CAUSE: {res['top_subsystem']}",
    }

    pulse = pulse_map.get(state, "")
    sub   = sub_map.get(state, "")

    # ── status hero ──────────────────────────
    st.markdown(f"""
    <div class="status-hero">
      <div class="status-icon" style="color:{sc['fg']};{pulse}">
        {res['status_icon']}
      </div>
      <div class="status-main" style="
           background:linear-gradient(135deg,{sc['fg']} 0%,#ff8c00 80%);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;filter:drop-shadow(0 0 18px {sc['glow']});">
        {state}
      </div>
      <div class="status-sub-label" style="color:{sc['fg']} !important;">{sub}</div>
      <span class="state-badge"
            style="background:{sc['bg']};border:1px solid {sc['border']};
                   color:{sc['fg']};">
        WINDOW #{res['window_idx']}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── score gauge + deviation bars ─────────
    col_gauge, col_dev = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="section-label">ISOLATION SCORE GAUGE</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_score_gauge(res["score"], res["threshold"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_dev:
        st.markdown('<div class="section-label">FEATURE DEVIATION BARS</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_deviation_bars(res["contributions"], res["norm_deviation"], res["n_feat"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── metrics row ──────────────────────────
    score_color = sc["fg"]
    top_f, top_c, top_d = res["top_features"][0]
    dir_arrow = "↑" if top_d > 0 else "↓"
    ratio_label = "ANOMALY" if res["is_anomaly"] else "NORMAL"

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, "IF ANOMALY SCORE",  f"{res['score']:+.4f}",
             "< THRESHOLD · ANOMALY" if res["is_anomaly"] else "> THRESHOLD · NORMAL"),
        (m2, "DECISION THRESHOLD", f"{res['threshold']:+.4f}", "decision_function"),
        (m3, "PRIMARY DEVIATION",  top_f.upper().replace("_", " "),
             f"{dir_arrow} {abs(top_d):.3f}σ FROM NORMAL"),
        (m4, "SUSPECT SUBSYSTEM",  res["top_subsystem"], "HIGHEST AGGREGATE DEV"),
    ]
    for col, label, val, unit in metrics:
        color = score_color if label in ("IF ANOMALY SCORE", "SUSPECT SUBSYSTEM") else "#ff8c00"
        with col:
            st.markdown(f"""
            <div class="metric-block">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color};font-size:1.05rem;">{val}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── top 3 ranked features ─────────────────
    st.markdown('<div class="section-label">TOP DEVIATING FEATURES — FORENSIC ATTRIBUTION</div>',
                unsafe_allow_html=True)
    rank_colors = [sc["fg"], "#ff8c00", "#ffd700"]
    r1, r2, r3  = st.columns(3)
    for ci, ((fname, fcontrib, fdev), col) in enumerate(
            zip(res["top_features"], [r1, r2, r3])):
        sys_label = next(
            (sn for sn, sf in SUBSYSTEMS.items() if fname in sf), "UNKNOWN"
        )
        dir_sym = "▲" if fdev > 0 else "▼"
        with col:
            st.markdown(f"""
            <div class="metric-block">
              <div class="metric-label">RANK #{ci+1}</div>
              <div class="metric-value" style="color:{rank_colors[ci]};font-size:0.95rem;">
                {fname.upper().replace("_"," ")}
              </div>
              <div class="metric-unit">{dir_sym} {abs(fdev):.3f}σ · {sys_label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── subsystem radar + decision space ─────
    ch1, ch2 = st.columns([2, 3])

    with ch1:
        st.markdown('<div class="section-label">SUBSYSTEM ATTRIBUTION RADAR</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_subsystem_radar(res["subsystem_scores"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with ch2:
        st.markdown('<div class="section-label">DECISION SPACE · FEATURE CONTRIBUTION LANDSCAPE</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_decision_space(res["contributions"], res["score"], res["threshold"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── component attribution summary ─────────
    st.markdown('<div class="section-label">COMPONENT ATTRIBUTION SUMMARY</div>',
                unsafe_allow_html=True)
    sorted_sys    = sorted(res["subsystem_scores"].items(), key=lambda x: x[1], reverse=True)
    max_sys_score = sorted_sys[0][1] if sorted_sys else 1

    cols = st.columns(len(sorted_sys))
    for (sname, sscore), col in zip(sorted_sys, cols):
        ratio     = sscore / (max_sys_score + 1e-9)
        intensity = "CRITICAL" if ratio > 0.8 else ("ELEVATED" if ratio > 0.45 else "NOMINAL")
        color     = SUBSYSTEM_COLORS.get(sname, "#ff8c00")
        with col:
            st.markdown(f"""
            <div class="metric-block" style="border-color:{color}22;">
              <div class="metric-label" style="color:{color}88 !important;">{sname}</div>
              <div class="metric-value" style="color:{color};font-size:1.1rem;">{sscore:.3f}</div>
              <div class="metric-unit">{intensity} · {ratio*100:.0f}% OF PEAK</div>
            </div>""", unsafe_allow_html=True)

    # ── footer ────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;margin-top:1rem;padding-top:0.8rem;
         border-top:1px solid rgba(255,140,0,0.06);">
      <span style="font-size:0.55rem;color:rgba(255,140,0,0.2);letter-spacing:0.2em;">
        SENTINELAI · FORENSIC ENGINE · ISOLATION FOREST
        &nbsp;·&nbsp; WINDOW {res["window_idx"]} / {data.shape[0]-1 if data is not None else "?"}
      </span>
    </div>
    """, unsafe_allow_html=True)