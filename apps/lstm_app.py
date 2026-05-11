"""
SentinelAI — LSTM Autoencoder Inference Surface
Headless Streamlit app, designed to be embedded in a React iframe.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI · Anomaly Scanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CENTRALIZED PATH MANAGEMENT
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.path_manager import PathManager

pm = PathManager()
pm.print_paths()  # Debug output at startup

SENSOR_NAMES = [
    "temperature_c", "vibration_mm_s", "pressure_kpa", "motor_rpm",
    "flow_rate_lpm", "power_consumption_kw", "coolant_temp_c",
    "acoustic_level_db", "oil_viscosity_cst", "humidity_pct", "ambient_temp_c",
]

N_CORE_SENSORS = len(SENSOR_NAMES)  # 11

# ─────────────────────────────────────────────
# CSS — CINEMATIC GLASSMORPHISM (matches RUL app)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── hide all Streamlit chrome ── */
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
        rgba(0,255,200,0.04) 0%,
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
    background: rgba(0,255,200,0.04);
    border: 1px solid rgba(0,255,200,0.18);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: rgba(0,255,200,0.6) !important;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00ffc8 !important;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.6rem;
    color: rgba(255,255,255,0.35) !important;
    margin-top: 0.15rem;
}

/* ── status hero ── */
.status-hero {
    text-align: center;
    padding: 1.4rem 0 0.6rem;
}
.status-icon {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    line-height: 1;
    filter: drop-shadow(0 0 24px rgba(0,255,200,0.5));
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
    letter-spacing: 0.22em;
    color: rgba(0,255,200,0.5) !important;
    text-transform: uppercase;
    margin-top: 0.35rem;
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

/* ── generate button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(0,255,200,0.12), rgba(0,120,255,0.12)) !important;
    border: 1px solid rgba(0,255,200,0.35) !important;
    color: #00ffc8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,200,0.22), rgba(0,120,255,0.22)) !important;
    box-shadow: 0 0 35px rgba(0,255,200,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── divider ── */
hr { border-color: rgba(0,255,200,0.08) !important; }
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,200,0.15), transparent);
    margin: 0.8rem 0;
}

/* ── section label ── */
.section-label {
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: rgba(0,255,200,0.4) !important;
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
    background: rgba(0,255,200,0.4);
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
    background: rgba(0,255,200,0.06);
    border: 1px solid rgba(0,255,200,0.15);
    border-radius: 999px;
    color: rgba(0,255,200,0.7) !important;
    letter-spacing: 0.08em;
}

/* ── load error card ── */
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
::-webkit-scrollbar-thumb { background: rgba(0,255,200,0.2); border-radius: 4px; }

/* ── pulse anim for anomaly ── */
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(255,59,92,0.6)); }
    50%       { filter: drop-shadow(0 0 28px rgba(255,59,92,0.95)); }
}
@keyframes pulse-warn {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(255,140,0,0.5)); }
    50%       { filter: drop-shadow(0 0 24px rgba(255,140,0,0.85)); }
}
@keyframes header-pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE  — unchanged
# ─────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim1, hidden_dim2, latent_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.latent_dim  = latent_dim
        self.enc_lstm1   = nn.LSTM(n_features,  hidden_dim1, batch_first=True)
        self.enc_lstm2   = nn.LSTM(hidden_dim1, latent_dim,  batch_first=True)
        self.dec_lstm1   = nn.LSTM(latent_dim,  latent_dim,  batch_first=True)
        self.dec_lstm2   = nn.LSTM(latent_dim,  hidden_dim1, batch_first=True)
        self.fc          = nn.Linear(hidden_dim1, n_features)

    def forward(self, x):
        out1, _       = self.enc_lstm1(x)
        out2, (h_n,_) = self.enc_lstm2(out1)
        latent        = h_n[-1].unsqueeze(1).repeat(1, self.window_size, 1)
        d1, _         = self.dec_lstm1(latent)
        d2, _         = self.dec_lstm2(d1)
        return self.fc(d2)


def detect_dims(state_dict):
    try:
        wih1       = state_dict["enc_lstm1.weight_ih_l0"]
        n_features = wih1.shape[1]
        hidden_dim1 = wih1.shape[0] // 4
        wih2       = state_dict["enc_lstm2.weight_ih_l0"]
        latent_dim = wih2.shape[0] // 4
        return n_features, hidden_dim1, hidden_dim1, latent_dim
    except KeyError:
        return 48, 64, 64, 32


# ─────────────────────────────────────────────
# CACHE — LOAD MODEL + DATA  — unchanged
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cpu")
    try:
        weights_path = pm.get_pt_path("lstm_ae", "lstm_ae_weights.pt")
        state_dict = torch.load(weights_path, map_location=device)
        n_features, hidden_dim1, hidden_dim2, latent_dim = detect_dims(state_dict)
        model = LSTMAutoencoder(
            n_features=n_features, hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2, latent_dim=latent_dim, window_size=50
        )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, n_features, hidden_dim1, latent_dim
    except Exception as e:
        sd_keys = list(state_dict.keys()) if 'state_dict' in locals() else 'N/A'
        weights_path = pm.get_pt_path("lstm_ae", "lstm_ae_weights.pt")
        st.error(f"Model loading failed: {str(e)}\nPath: {weights_path}\nState dict keys: {sd_keys}")
        raise


@st.cache_resource(show_spinner=False)
def load_data():
    try:
        return np.load(pm.get_data_path("X_test_all.npy")).astype(np.float32)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        raise


@st.cache_resource(show_spinner=False)
def load_threshold():
    try:
        with open(pm.get_pkl_path("lstm_ae", "lstm_ae_threshold.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load threshold: {e}")
        raise


# ─────────────────────────────────────────────
# INFERENCE  — unchanged
# ─────────────────────────────────────────────
def run_inference(window_np, model, threshold):
    device = torch.device("cpu")
    x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = model(x)
    recon_np          = recon.squeeze(0).numpy()
    per_feature_error = np.mean((window_np - recon_np) ** 2, axis=0)
    rec_error         = float(np.mean(per_feature_error))

    ratio = rec_error / (threshold + 1e-9)
    if ratio < 0.6:
        confidence = (1 - ratio / 0.6) * 100
        state      = "HEALTHY"
    elif ratio < 1.0:
        confidence = (ratio - 0.6) / 0.4 * 100
        state      = "WARNING"
    else:
        confidence = min((ratio - 1.0) * 60 + 30, 99)
        state      = "ANOMALY DETECTED"

    core_errors = per_feature_error[:N_CORE_SENSORS]
    top_idx     = np.argsort(core_errors)[::-1][:3]
    top_sensors = [(SENSOR_NAMES[i], float(core_errors[i])) for i in top_idx]

    return {
        "rec_error":         rec_error,
        "per_feature_error": per_feature_error,
        "state":             state,
        "confidence":        confidence,
        "top_sensors":       top_sensors,
        "recon":             recon_np,
        "original":          window_np,
    }


# ─────────────────────────────────────────────
# PLOTLY THEME — cinematic transparent
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="JetBrains Mono, Courier New, monospace", size=10),
    margin=dict(l=40, r=16, t=28, b=32),
    xaxis=dict(
        gridcolor="rgba(0,255,200,0.07)",
        zerolinecolor="rgba(0,255,200,0.12)",
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9),
    ),
    yaxis=dict(
        gridcolor="rgba(0,255,200,0.07)",
        zerolinecolor="rgba(0,255,200,0.12)",
        linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=9),
    ),
)

STATE_COLORS = {
    "HEALTHY":          {"fg": "#00ffc8", "bg": "rgba(0,255,200,0.08)",
                         "border": "rgba(0,255,200,0.25)", "glow": "rgba(0,255,200,0.4)"},
    "WARNING":          {"fg": "#ffb400", "bg": "rgba(255,180,0,0.07)",
                         "border": "rgba(255,180,0,0.3)",  "glow": "rgba(255,180,0,0.4)"},
    "ANOMALY DETECTED": {"fg": "#ff3b5c", "bg": "rgba(255,59,92,0.07)",
                         "border": "rgba(255,59,92,0.3)",  "glow": "rgba(255,59,92,0.45)"},
}


def _base_layout(**overrides):
    layout = dict(PLOTLY_BASE)
    layout.update(overrides)
    return layout


def make_recon_error_chart(per_feature_error, n_features):
    labels = SENSOR_NAMES + [f"feat_{i}" for i in range(N_CORE_SENSORS, n_features)]
    colors = ["#ff3b5c" if i < N_CORE_SENSORS else "#00ffc8" for i in range(n_features)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(n_features)),
        y=per_feature_error,
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=labels,
        hovertemplate="<b>%{text}</b><br>Error: %{y:.6f}<extra></extra>",
        name="Reconstruction Error",
    ))

    layout = _base_layout(
        height=190,
        showlegend=False,
        title=dict(
            text="RECONSTRUCTION ERROR PROFILE",
            font=dict(size=9, color="rgba(0,255,200,0.4)"),
            x=0.01,
        ),
        bargap=0.15,
    )
    fig.update_layout(**layout)
    return fig


def make_waveform_chart(original, recon, sensor_idx=0):
    t    = np.arange(original.shape[0])
    orig = original[:, sensor_idx]
    rec  = recon[:, sensor_idx]

    fig = go.Figure()
    # Fill between original and reconstruction
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([orig, rec[::-1]]),
        fill="toself",
        fillcolor="rgba(255,59,92,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=orig,
        mode="lines",
        line=dict(color="#00ffc8", width=1.8),
        name="ORIGINAL",
        hovertemplate="t=%{x}  val=%{y:.4f}<extra>Original</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=rec,
        mode="lines",
        line=dict(color="#ff8c00", width=1.3, dash="dot"),
        name="RECONSTRUCTED",
        hovertemplate="t=%{x}  val=%{y:.4f}<extra>Reconstructed</extra>",
    ))

    layout = _base_layout(
        height=210,
        title=dict(
            text=f"SIGNAL WAVEFORM — {SENSOR_NAMES[sensor_idx].upper()}",
            font=dict(size=9, color="rgba(0,255,200,0.4)"),
            x=0.01,
        ),
        legend=dict(
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
            orientation="h", y=1.14,
        ),
    )
    fig.update_layout(**layout)
    return fig


def make_radar_chart(per_feature_error):
    labels = SENSOR_NAMES
    values = list(per_feature_error[:N_CORE_SENSORS])
    max_v  = max(values) if max(values) > 0 else 1
    norm   = [v / max_v for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm + [norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(0,255,200,0.06)",
        line=dict(color="#00ffc8", width=1.6),
        mode="lines+markers",
        marker=dict(size=5, color="#00ffc8"),
        name="Error Profile",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Courier New, monospace",
                  color="#94a3b8", size=9),
        margin=dict(l=40, r=40, t=36, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(0,255,200,0.08)",
                linecolor="rgba(0,255,200,0.12)",
                tickfont=dict(size=8),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor="rgba(0,255,200,0.08)",
                linecolor="rgba(0,255,200,0.15)",
                tickfont=dict(size=8, color="#94a3b8"),
            ),
        ),
        title=dict(
            text="SENSOR ANOMALY RADAR",
            font=dict(size=9, color="rgba(0,255,200,0.4)"),
            x=0.01,
        ),
        showlegend=False,
        height=250,
    )
    return fig


def make_confidence_gauge(state, confidence):
    sc     = STATE_COLORS.get(state, STATE_COLORS["HEALTHY"])
    max_v  = 145  # reuse same gauge scale feel as RUL app

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number=dict(
            font=dict(size=32, color=sc["fg"],
                      family="JetBrains Mono, Courier New, monospace"),
            suffix="%",
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickfont=dict(color="rgba(255,255,255,0.3)", size=8),
                tickcolor="rgba(255,255,255,0.15)",
                tickvals=[0, 25, 50, 75, 100],
            ),
            bar=dict(color=sc["fg"], thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0,   33],  color="rgba(255,59,92,0.07)"),
                dict(range=[33,  66],  color="rgba(255,180,0,0.05)"),
                dict(range=[66, 100],  color="rgba(0,255,200,0.05)"),
            ],
            threshold=dict(
                line=dict(color=sc["fg"], width=2),
                thickness=0.75,
                value=confidence,
            ),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    layout = dict(PLOTLY_BASE)
    layout["margin"] = dict(l=20, r=20, t=10, b=10)
    layout["height"]  = 220
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng()

# ─────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.4rem 0.2rem 0.8rem;
            border-bottom:1px solid rgba(0,255,200,0.08);
            margin-bottom:0.8rem;">
  <div style="display:flex;align-items:center;gap:0.7rem;">
    <div style="width:8px;height:8px;border-radius:50%;background:#00ffc8;
                box-shadow:0 0 10px #00ffc8;
                animation:header-pulse 1.5s ease-in-out infinite;"></div>
    <span style="font-size:0.65rem;letter-spacing:0.25em;
                 color:rgba(0,255,200,0.7);text-transform:uppercase;">
      SentinelAI · Anomaly Scanner
    </span>
  </div>
  <div style="display:flex;gap:1.2rem;">
    <span style="font-size:0.58rem;color:rgba(255,255,255,0.25);">LSTM AUTOENCODER</span>
    <span style="font-size:0.58rem;color:rgba(255,255,255,0.25);">REAL-TIME INFERENCE</span>
    <span style="font-size:0.58rem;color:rgba(0,255,200,0.4);">TATA TECH · JAMSHEDPUR</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
load_error = None
model = data = threshold = n_features = None

try:
    model, n_features, hidden_dim, latent_dim = load_model()
    data      = load_data()
    threshold = load_threshold()
    if isinstance(threshold, dict):
        threshold = threshold.get("global", list(threshold.values())[0])
    threshold = float(threshold)
except Exception as e:
    load_error = str(e)

# ─────────────────────────────────────────────
# MODEL STATUS PILLS + GENERATE BUTTON
# ─────────────────────────────────────────────
if not load_error and data is not None:
    n_windows = data.shape[0]
    st.markdown(f"""
    <div class="stat-row">
      <span class="stat-pill">WINDOW 50 CYCLES</span>
      <span class="stat-pill">{data.shape[2]} FEATURES</span>
      <span class="stat-pill">{N_CORE_SENSORS} CORE SENSORS</span>
      <span class="stat-pill">{n_windows:,} WINDOWS</span>
      <span class="stat-pill">THRESHOLD {threshold:.5f}</span>
      <span class="stat-pill">LIVE INFERENCE</span>
    </div>
    """, unsafe_allow_html=True)

gen_col, info_col = st.columns([1, 3])
with gen_col:
    run = st.button("⟳  GENERATE SCENARIO", use_container_width=True)

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
# RUN INFERENCE  — unchanged logic
# ─────────────────────────────────────────────
if run and model is not None and data is not None:
    idx    = st.session_state.rng.integers(0, data.shape[0])
    window = data[idx]
    result = run_inference(window, model, threshold)
    result["window_idx"] = int(idx)
    st.session_state.result = result

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# OUTPUT DISPLAY
# ─────────────────────────────────────────────
res = st.session_state.result

if res is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;opacity:0.4;">
      <div style="font-size:3rem;margin-bottom:1rem;
                  filter:drop-shadow(0 0 20px rgba(0,255,200,0.3));">◎</div>
      <div style="font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;
                  color:rgba(0,255,200,0.6);">Anomaly Scanner Standby</div>
      <div style="font-size:0.55rem;color:rgba(255,255,255,0.25);
                  margin-top:0.5rem;letter-spacing:0.1em;">
        Press GENERATE SCENARIO to run live inference
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    state = res["state"]
    sc    = STATE_COLORS.get(state, STATE_COLORS["HEALTHY"])

    # ── icon + headline maps ──────────────────
    icon_map = {
        "HEALTHY":          "◉",
        "WARNING":          "◈",
        "ANOMALY DETECTED": "⬟",
    }
    pulse_map = {
        "HEALTHY":          "",
        "WARNING":          "animation:pulse-warn 2s ease-in-out infinite;",
        "ANOMALY DETECTED": "animation:pulse-glow 1.8s ease-in-out infinite;",
    }
    sub_map = {
        "HEALTHY":          "ALL SYSTEMS NOMINAL · RECONSTRUCTION WITHIN BOUNDS",
        "WARNING":          "DEGRADATION DETECTED · MONITORING ESCALATED",
        "ANOMALY DETECTED": "ANOMALY CONFIRMED · RECONSTRUCTION BOUNDARY EXCEEDED",
    }

    icon  = icon_map.get(state, "◉")
    pulse = pulse_map.get(state, "")
    sub   = sub_map.get(state, "")

    # ── status hero ──────────────────────────
    st.markdown(f"""
    <div class="status-hero">
      <div class="status-icon" style="color:{sc['fg']};{pulse}">{icon}</div>
      <div class="status-main" style="
           background:linear-gradient(135deg,{sc['fg']} 0%,#00b4ff 80%);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;filter:drop-shadow(0 0 18px {sc['glow']});">
        {state}
      </div>
      <div class="status-sub-label">{sub}</div>
      <span class="state-badge"
            style="background:{sc['bg']};border:1px solid {sc['border']};
                   color:{sc['fg']};">
        WINDOW #{res['window_idx']}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── metrics row ──────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    ratio = res["rec_error"] / (threshold + 1e-9)
    top_name, top_err = res["top_sensors"][0]

    metrics = [
        (m1, "RECON ERROR",     f"{res['rec_error']:.6f}", "GLOBAL MSE"),
        (m2, "THRESHOLD",       f"{threshold:.6f}",
                                "↑ EXCEEDED" if res["rec_error"] > threshold else "↓ NOMINAL"),
        (m3, "ERROR RATIO",     f"{ratio:.3f}×",
                                "CRITICAL" if ratio > 1.5 else "NOMINAL"),
        (m4, "TOP SENSOR",      top_name.upper().replace("_"," "), f"err {top_err:.6f}"),
    ]
    for col, label, val, unit in metrics:
        color = sc["fg"] if label in ("RECON ERROR", "TOP SENSOR") else "#00ffc8"
        with col:
            st.markdown(f"""
            <div class="metric-block">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color};font-size:1.1rem;">{val}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── confidence gauge + waveform ──────────
    col_gauge, col_wave = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="section-label">ANOMALY CONFIDENCE</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_confidence_gauge(state, res["confidence"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_wave:
        top_sensor_idx = SENSOR_NAMES.index(res["top_sensors"][0][0])
        st.markdown('<div class="section-label">SIGNAL WAVEFORM · ORIGINAL vs RECONSTRUCTED</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_waveform_chart(res["original"], res["recon"], sensor_idx=top_sensor_idx),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── top sensors ranked ───────────────────
    st.markdown('<div class="section-label">TOP ANOMALOUS SENSORS</div>',
                unsafe_allow_html=True)
    rank_colors = [sc["fg"], "#ff8c00", "#ffd700"]
    r1, r2, r3  = st.columns(3)
    for ci, ((sname, serr), col) in enumerate(zip(res["top_sensors"], [r1, r2, r3])):
        with col:
            st.markdown(f"""
            <div class="metric-block">
              <div class="metric-label">RANK #{ci+1} SENSOR</div>
              <div class="metric-value" style="color:{rank_colors[ci]};font-size:1rem;">
                {sname.upper().replace("_"," ")}
              </div>
              <div class="metric-unit">ERROR {serr:.6f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── error profile bar chart ───────────────
    st.markdown('<div class="section-label">RECONSTRUCTION ERROR PROFILE — ALL FEATURES</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        make_recon_error_chart(res["per_feature_error"], n_features),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── radar chart ───────────────────────────
    st.markdown('<div class="section-label">SENSOR ANOMALY RADAR</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        make_radar_chart(res["per_feature_error"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── footer ────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;margin-top:1rem;padding-top:0.8rem;
         border-top:1px solid rgba(0,255,200,0.06);">
      <span style="font-size:0.55rem;color:rgba(0,255,200,0.2);letter-spacing:0.2em;">
        SENTINELAI · TATA TECHNOLOGIES · JAMSHEDPUR PREDICTIVE MAINTENANCE
        &nbsp;·&nbsp; WINDOW {res["window_idx"]} / {data.shape[0]-1 if data is not None else "?"}
      </span>
    </div>
    """, unsafe_allow_html=True)