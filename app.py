"""
app.py — bpdisdet
══════════════════════════════════════════════════════════════════════
Multimodal Mental Health Screening Tool
Bipolar Disorder Detection via Facial Affect & Linguistic Analysis

SDG 3: Good Health and Well-Being — Equitable Early Diagnostics
"""

import os
import sys
import time
import json
import base64
import tempfile
from io import BytesIO
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config — must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="bpdisdet · Mental Health Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

# ── CSS Theming ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2230;
    --bg-glass:      rgba(28, 34, 48, 0.85);
    --accent-teal:   #2dd4bf;
    --accent-amber:  #f59e0b;
    --accent-rose:   #f43f5e;
    --accent-blue:   #60a5fa;
    --accent-purple: #a78bfa;
    --text-primary:  #e8edf5;
    --text-muted:    #8b9ab0;
    --border:        rgba(255,255,255,0.07);
    --shadow-glow:   0 0 30px rgba(45, 212, 191, 0.08);
    --font-display:  'DM Serif Display', Georgia, serif;
    --font-body:     'DM Sans', system-ui, sans-serif;
    --font-mono:     'DM Mono', 'Courier New', monospace;
  }

  /* Reset */
  html, body, [class*="css"], .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
  }

  /* App background */
  .stApp { background: var(--bg-primary); }

  /* Top gradient bar */
  .stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue), var(--accent-purple));
    z-index: 9999;
  }

  /* Main header */
  .bpd-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
  }
  .bpd-hero h1 {
    font-family: var(--font-display) !important;
    font-size: clamp(2rem, 4vw, 3.2rem) !important;
    font-weight: 400 !important;
    color: var(--accent-teal) !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
  }
  .bpd-hero .subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .sdg-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a6b3c, #0f4d2c);
    border: 1px solid rgba(74, 222, 128, 0.3);
    color: #4ade80;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
    font-family: var(--font-mono);
  }

  /* Cards */
  .bpd-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-glow);
    transition: border-color 0.2s;
  }
  .bpd-card:hover { border-color: rgba(45, 212, 191, 0.2); }
  .bpd-card h3 {
    font-family: var(--font-display) !important;
    font-size: 1.15rem !important;
    color: var(--accent-teal) !important;
    margin-bottom: 0.6rem !important;
  }

  /* Risk chips */
  .risk-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
  }
  .risk-high     { background: rgba(244,63,94,0.18); color: #f87171; border: 1px solid rgba(244,63,94,0.35); }
  .risk-moderate { background: rgba(245,158,11,0.18); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
  .risk-low      { background: rgba(45,212,191,0.15); color: #2dd4bf; border: 1px solid rgba(45,212,191,0.3); }
  .risk-minimal  { background: rgba(96,165,250,0.15); color: #93c5fd; border: 1px solid rgba(96,165,250,0.3); }

  /* Metric tiles */
  .metric-tile {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-tile .value {
    font-family: var(--font-mono);
    font-size: 2rem;
    font-weight: 500;
    line-height: 1.1;
  }
  .metric-tile .label {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    padding: 8px 16px !important;
    border: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-teal) !important;
    border: 1px solid rgba(45,212,191,0.2) !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

  /* Inputs */
  .stTextArea textarea, .stTextInput input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(45,212,191,0.4) !important;
    box-shadow: 0 0 0 2px rgba(45,212,191,0.1) !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent-teal), #0891b2) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(45,212,191,0.3) !important;
  }
  .stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--font-display) !important;
    color: var(--accent-teal) !important;
  }

  /* Alerts */
  .stAlert {
    border-radius: 8px !important;
    border-left-width: 3px !important;
  }

  /* Slider */
  .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: var(--accent-teal) !important;
  }

  /* Progress bars */
  .stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
    border-radius: 4px !important;
  }

  /* Disclaimer banner */
  .disclaimer-banner {
    background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    font-size: 0.83rem;
    color: #fbbf24;
    line-height: 1.5;
  }

  /* Webcam container */
  .webcam-container {
    border: 2px dashed rgba(45,212,191,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    background: rgba(13,17,23,0.6);
  }

  /* Score bars */
  .score-row { margin: 6px 0; }
  .score-label { font-size: 0.82rem; color: var(--text-muted); margin-bottom: 2px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(45,212,191,0.3); }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "facial_session": None,
        "text_results": [],
        "screening_result": None,
        "camera_running": False,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "patient_name": "",
        "patient_age": 25,
        "patient_gender": "Not specified",
        "capture_frames": [],
        "analysis_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 bpdisdet")
    st.markdown("<span style='color:#8b9ab0;font-size:0.8rem'>Bipolar Disorder Detection</span>", unsafe_allow_html=True)
    st.divider()

    # API Key
    st.markdown("**⚙️ Configuration**")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        help="Provide your Anthropic API key for enhanced LLM-based text analysis. "
             "Without it, a local heuristic analyser will be used.",
        placeholder="sk-ant-..."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✓ API key configured", icon="🔐")
    else:
        st.info("💡 No API key — using local heuristic analyser", icon="ℹ️")

    st.divider()

    # Patient Info
    st.markdown("**👤 Patient Information**")
    st.session_state.patient_name = st.text_input(
        "Name / Alias", value=st.session_state.patient_name, placeholder="Anonymous"
    )
    st.session_state.patient_age = st.slider("Age", 12, 90, st.session_state.patient_age)
    st.session_state.patient_gender = st.selectbox(
        "Gender", ["Not specified", "Female", "Male", "Non-binary", "Prefer not to say"]
    )
    clinician = st.text_input("Clinician / Reviewer", placeholder="Dr. ...")

    st.divider()

    # About
    st.markdown("**ℹ️ About**")
    st.markdown(
        "<div style='font-size:0.78rem;color:#8b9ab0;line-height:1.6'>"
        "bpdisdet uses facial affect analysis (OpenCV + DeepFace) and "
        "LLM-powered linguistic screening to identify potential markers of "
        "bipolar spectrum disorders.<br><br>"
        "<b style='color:#4ade80'>SDG 3:</b> Good Health & Well-Being — "
        "equitable mental health access in low-resource settings."
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#6b7280;text-align:center'>"
        "v1.0 · For research & screening only<br>Not a diagnostic instrument"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bpd-hero">
  <h1>bpdisdet</h1>
  <div class="subtitle">Multimodal Bipolar Spectrum · Early Screening System</div>
  <div class="sdg-badge">✦ UN SDG-3 · Mental Health Access · v1.0</div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-banner">
  ⚠ <b>Medical Disclaimer:</b> This tool provides <i>screening support only</i> and does not constitute
  a clinical diagnosis. All results must be reviewed by a licensed mental health professional.
  If you are in crisis, contact a helpline immediately:
  <b>iCall (India): 9152987821</b> · <b>Vandrevala: 1860-2662-345</b> · <b>Emergency: 112</b>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📝  Text Analysis",
    "📷  Facial Affect",
    "📊  Screening Results",
    "📄  Report & Export",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · TEXT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="bpd-card"><h3>🔤 Linguistic Marker Analysis</h3>'
                '<p style="color:#8b9ab0;font-size:0.85rem;margin:0">Submit free-text entries for psycholinguistic analysis. '
                'Multiple entries simulate a longitudinal mood journal.</p></div>',
                unsafe_allow_html=True)

    col_input, col_guide = st.columns([3, 1])

    with col_guide:
        st.markdown("""
<div class="bpd-card">
<h3 style="font-size:0.95rem !important">💡 Writing Prompts</h3>
<div style="font-size:0.8rem;color:#8b9ab0;line-height:1.8">

<b style="color:#e8edf5">Describe:</b><br>
• How you've felt this week<br>
• Your sleep & energy levels<br>
• Plans or goals you have<br>
• Things you enjoyed / avoided<br>
• Any worries or fears<br>
• Thoughts racing or slowing<br><br>

<b style="color:#e8edf5">The more detail, the better.</b><br>
<span style="color:#4ade80">∼50–200 words recommended.</span>

</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="bpd-card" style="margin-top:0.5rem">
<h3 style="font-size:0.95rem !important">📌 Markers Detected</h3>
<div style="font-size:0.77rem;color:#8b9ab0;line-height:1.8">
<b style="color:#f43f5e">Mania:</b><br>
Pressured speech · Grandiosity<br>
Flight of ideas · Decreased sleep<br><br>
<b style="color:#60a5fa">Depression:</b><br>
Anhedonia · Hopelessness<br>
Psychomotor slowing · Fatigue<br><br>
<b style="color:#a78bfa">Mixed:</b><br>
Dysphoric arousal · Agitation<br>
Racing negative thoughts
</div>
</div>
        """, unsafe_allow_html=True)

    with col_input:
        # Entry tabs
        entry_mode = st.radio(
            "Entry mode",
            ["Single entry", "Multi-entry journal (3 days)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        entries = []

        if entry_mode == "Single entry":
            text_in = st.text_area(
                "Your thoughts / journal entry",
                height=200,
                placeholder="Describe how you've been feeling lately. Be as open and detailed as you like. "
                            "This is a safe, confidential space...",
                label_visibility="collapsed",
            )
            if text_in.strip():
                entries = [text_in]
        else:
            st.markdown("<div style='color:#8b9ab0;font-size:0.82rem;margin-bottom:0.5rem'>"
                        "Enter one journal entry per day (past 3 days). Leave empty if not available.</div>",
                        unsafe_allow_html=True)
            for d in range(1, 4):
                e = st.text_area(
                    f"Day {d} entry",
                    height=120,
                    placeholder=f"How were you feeling {['today', 'yesterday', '2 days ago'][d-1]}?",
                    key=f"entry_day_{d}",
                )
                if e.strip():
                    entries.append(e)

        # Sample entries for demo
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📋 Demo: Manic", use_container_width=True):
                st.session_state["demo_text"] = (
                    "I have SO many ideas right now I can barely sleep! I don't even NEED sleep — "
                    "I feel more powerful and alive than ever before. I started three new businesses "
                    "this week and I know at least two of them will make me a millionaire. "
                    "People just can't keep up with my mind, it moves so fast. I've been talking to "
                    "investors, writing my book, and planning a world tour — all simultaneously!! "
                    "Everyone thinks I'm special and they're RIGHT. I was meant for greatness."
                )
        with col_b:
            if st.button("📋 Demo: Depressed", use_container_width=True):
                st.session_state["demo_text"] = (
                    "I don't really see the point of much anymore. Everything feels heavy and grey. "
                    "I haven't left the house in days... maybe a week? I can't remember. Sleep is "
                    "all I want to do but even that doesn't help. I used to enjoy painting but now "
                    "the brushes just sit there. I feel worthless. Like a burden to everyone around me. "
                    "My thoughts are so slow, like wading through mud. Nothing will get better. "
                    "I just feel so empty and tired all the time."
                )
        with col_c:
            if st.button("📋 Demo: Mixed", use_container_width=True):
                st.session_state["demo_text"] = (
                    "My mind won't stop racing but everything I think about fills me with dread. "
                    "I'm furious at everyone for no reason then crying an hour later. "
                    "I have this horrible energy — restless, agitated — but I hate it. "
                    "I feel like I need to do something but every idea feels pointless. "
                    "Can't sleep but I'm exhausted. Started 4 projects and abandoned all of them "
                    "the same day. I feel like I'm going to explode and disappear at the same time."
                )

        # Show demo text if loaded
        if "demo_text" in st.session_state and st.session_state.demo_text:
            st.text_area(
                "Loaded demo text (copy above)",
                value=st.session_state.demo_text,
                height=100,
                disabled=True,
                key="demo_display",
            )
            if not entries:
                entries = [st.session_state.demo_text]

        # Analyse button
        st.markdown("<br>", unsafe_allow_html=True)
        run_text = st.button("🔍  Analyse Text", use_container_width=True, key="run_text_btn")

        if run_text:
            if not entries:
                st.warning("Please enter at least one text entry before analysing.")
            else:
                with st.spinner("Performing psycholinguistic analysis..."):
                    from modules.text_analysis import analyse_with_api, analyse_heuristic

                    results = []
                    api_key = st.session_state.api_key

                    progress_bar = st.progress(0)
                    for i, entry in enumerate(entries):
                        if api_key:
                            res = analyse_with_api(entry, api_key)
                        else:
                            res = analyse_heuristic(entry)
                        results.append(res)
                        progress_bar.progress((i + 1) / len(entries))

                    st.session_state.text_results = results
                    progress_bar.empty()

                    st.success(f"✓ Analysed {len(results)} entr{'y' if len(results)==1 else 'ies'} "
                               f"using {results[0].analysis_method.upper()} method.")

    # ── Results display ──────────────────────────────────────────────────────
    if st.session_state.text_results:
        results = st.session_state.text_results
        st.divider()
        st.markdown("### 📊 Text Analysis Results")

        # Aggregate
        avg_mania = sum(r.mania_score for r in results) / len(results)
        avg_depr  = sum(r.depression_score for r in results) / len(results)
        avg_mixed = sum(r.mixed_score for r in results) / len(results)
        latest    = results[-1]

        # Score tiles
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            color = "#f87171" if avg_mania > 65 else "#fbbf24" if avg_mania > 35 else "#2dd4bf"
            st.markdown(f"""<div class="metric-tile">
                <div class="value" style="color:{color}">{avg_mania:.0f}</div>
                <div class="label">Mania Index</div></div>""", unsafe_allow_html=True)
        with mc2:
            color = "#f87171" if avg_depr > 65 else "#fbbf24" if avg_depr > 35 else "#60a5fa"
            st.markdown(f"""<div class="metric-tile">
                <div class="value" style="color:{color}">{avg_depr:.0f}</div>
                <div class="label">Depression Index</div></div>""", unsafe_allow_html=True)
        with mc3:
            color = "#a78bfa" if avg_mixed > 40 else "#8b9ab0"
            st.markdown(f"""<div class="metric-tile">
                <div class="value" style="color:{color}">{avg_mixed:.0f}</div>
                <div class="label">Mixed State</div></div>""", unsafe_allow_html=True)
        with mc4:
            risk_cls = f"risk-{latest.risk_level}"
            st.markdown(f"""<div class="metric-tile">
                <div class="value"><span class="risk-chip {risk_cls}">{latest.risk_level.upper()}</span></div>
                <div class="label" style="margin-top:8px">Risk Level</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        ch1, ch2 = st.columns(2)

        with ch1:
            # Radar chart of linguistic markers
            m = latest.markers
            cats = ["Pressured Speech", "Flight of Ideas", "Grandiosity",
                    "Anhedonia", "Hopelessness", "Psychomotor Slow",
                    "Irritability", "Mixed Dysphoria"]
            vals = [m.pressured_speech, m.flight_of_ideas, m.grandiosity,
                    m.anhedonia_markers, m.hopelessness, m.psychomotor_slowdown,
                    m.irritability, m.mixed_dysphoria]

            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(45,212,191,0.12)",
                line=dict(color="#2dd4bf", width=2),
                marker=dict(color="#2dd4bf", size=5),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(22,27,34,0.8)",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="rgba(255,255,255,0.06)",
                                   tickfont=dict(color="#8b9ab0", size=9)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                     tickfont=dict(color="#8b9ab0", size=10))
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#8b9ab0"),
                title=dict(text="Linguistic Marker Profile", font=dict(color="#e8edf5", size=13)),
                margin=dict(l=30, r=30, t=50, b=20),
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with ch2:
            if len(results) > 1:
                # Longitudinal trend
                fig_trend = go.Figure()
                x_labels = [f"Entry {i+1}" for i in range(len(results))]
                fig_trend.add_trace(go.Scatter(
                    x=x_labels, y=[r.mania_score for r in results],
                    name="Mania", line=dict(color="#f87171", width=2),
                    mode="lines+markers", marker=dict(size=7)))
                fig_trend.add_trace(go.Scatter(
                    x=x_labels, y=[r.depression_score for r in results],
                    name="Depression", line=dict(color="#60a5fa", width=2),
                    mode="lines+markers", marker=dict(size=7)))
                fig_trend.add_trace(go.Scatter(
                    x=x_labels, y=[r.mixed_score for r in results],
                    name="Mixed", line=dict(color="#a78bfa", width=2),
                    mode="lines+markers", marker=dict(size=7)))
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
                    font=dict(family="DM Sans", color="#8b9ab0"),
                    title=dict(text="Longitudinal Score Trend", font=dict(color="#e8edf5", size=13)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=50, b=20), height=320,
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                # Bar chart of all scores
                fig_bar = go.Figure(go.Bar(
                    x=["Mania", "Depression", "Mixed State", "Confidence"],
                    y=[latest.mania_score, latest.depression_score, latest.mixed_score, latest.confidence],
                    marker_color=["#f87171", "#60a5fa", "#a78bfa", "#2dd4bf"],
                    text=[f"{v:.0f}" for v in [latest.mania_score, latest.depression_score,
                                               latest.mixed_score, latest.confidence]],
                    textposition="outside", textfont=dict(color="#e8edf5"),
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
                    font=dict(family="DM Sans", color="#8b9ab0"),
                    title=dict(text="Score Overview", font=dict(color="#e8edf5", size=13)),
                    yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0)",),
                    margin=dict(l=10, r=10, t=50, b=20), height=320,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Clinical summary
        if latest.clinical_summary:
            st.markdown(f"""
<div class="bpd-card" style="border-color:rgba(45,212,191,0.15)">
<h3>🩺 Clinical Summary</h3>
<p style="color:#b8c4d4;font-size:0.9rem;line-height:1.7">{latest.clinical_summary}</p>
</div>""", unsafe_allow_html=True)

        # Key phrases
        if latest.key_phrases:
            phrases_html = " ".join(
                f'<span style="background:rgba(244,63,94,0.15);color:#f87171;border:1px solid rgba(244,63,94,0.3);'
                f'padding:3px 10px;border-radius:20px;font-size:0.78rem;margin:2px;display:inline-block">{p}</span>'
                if p.startswith("⚠") else
                f'<span style="background:rgba(45,212,191,0.12);color:#2dd4bf;border:1px solid rgba(45,212,191,0.25);'
                f'padding:3px 10px;border-radius:20px;font-size:0.78rem;margin:2px;display:inline-block">{p}</span>'
                for p in latest.key_phrases
            )
            st.markdown(f"""<div class="bpd-card">
<h3>🏷️ Key Linguistic Markers</h3>
{phrases_html}
</div>""", unsafe_allow_html=True)

        # Red flags
        for r in results:
            for rec in r.recommendations:
                if "URGENT" in rec or "crisis" in rec.lower():
                    st.error(f"🔴 {rec}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · FACIAL AFFECT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="bpd-card"><h3>📷 Real-time Facial Affect Analysis</h3>'
                '<p style="color:#8b9ab0;font-size:0.85rem;margin:0">Uses OpenCV + DeepFace to detect facial emotions '
                'and compute affective instability metrics over time.</p></div>',
                unsafe_allow_html=True)

    cam_col, info_col = st.columns([3, 2])

    with info_col:
        st.markdown("""
<div class="bpd-card">
<h3>📌 How It Works</h3>
<div style="font-size:0.82rem;color:#8b9ab0;line-height:1.9">

<b style="color:#e8edf5">1. Face Detection</b><br>
OpenCV Haar cascade detects<br>facial regions in real time<br><br>

<b style="color:#e8edf5">2. Emotion Classification</b><br>
DeepFace (FER+ model) classifies<br>
7 basic emotions per frame<br><br>

<b style="color:#e8edf5">3. Affective Instability</b><br>
MSSD (Mean Square of Successive<br>
Differences) on valence timeline<br><br>

<b style="color:#e8edf5">4. Pattern Detection</b><br>
Manic / depressive / mixed patterns<br>
identified from emotion clusters

</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="bpd-card">
<h3>🎭 Emotion → Affect Map</h3>
<div style="font-size:0.8rem;line-height:2.1">
<span style="color:#4ade80">● Happy</span> — elevated valence<br>
<span style="color:#fbbf24">● Surprise</span> — high arousal<br>
<span style="color:#f87171">● Angry</span> — manic/mixed marker<br>
<span style="color:#60a5fa">● Sad</span> — low valence<br>
<span style="color:#a78bfa">● Fear</span> — mixed state<br>
<span style="color:#fb923c">● Disgust</span> — depressive marker<br>
<span style="color:#8b9ab0">● Neutral</span> — baseline
</div>
</div>
        """, unsafe_allow_html=True)

    with cam_col:
        st.markdown("#### 🎯 Capture facial emotions for analysis")

        cam_mode = st.radio(
            "Capture mode",
            ["📁 Upload image/video", "📸 Live webcam capture"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if "📁" in cam_mode:
            uploaded = st.file_uploader(
                "Upload a photo or short video clip",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                label_visibility="collapsed",
            )

            if uploaded:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)

                if uploaded.type.startswith("image"):
                    import cv2
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        from modules.facial_analysis import (
                            FaceDetector, FacialSession, EmotionFrame,
                            analyse_frame, compute_session_metrics
                        )
                        detector = FaceDetector()
                        annotated, ef = analyse_frame(img, detector)

                        # Display
                        _, buf = cv2.imencode(".jpg", annotated)
                        st.image(buf.tobytes(), channels="BGR", use_container_width=True)

                        if ef:
                            session = FacialSession()
                            # Simulate brief session from single frame
                            for _ in range(8):
                                session.frames.append(ef)
                            session = compute_session_metrics(session)
                            st.session_state.facial_session = session

                            # Show emotion scores
                            st.markdown("**Detected emotions:**")
                            sorted_emotions = sorted(ef.scores.items(), key=lambda x: -x[1])
                            for emo, score in sorted_emotions[:4]:
                                st.progress(int(score), text=f"{emo.capitalize()}: {score:.1f}%")

                            st.success(f"✓ Dominant: **{ef.dominant.upper()}** | "
                                       f"Valence: {ef.valence:+.2f} | Arousal: {ef.arousal:+.2f}")
                        else:
                            st.warning("No face detected in the uploaded image.")
                    else:
                        st.error("Could not decode image.")

                elif uploaded.type.startswith("video"):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(file_bytes.tobytes())
                        tmp_path = tmp.name

                    with st.spinner("Analysing video frames..."):
                        import cv2
                        from modules.facial_analysis import (
                            FaceDetector, FacialSession,
                            analyse_frame, compute_session_metrics
                        )
                        cap = cv2.VideoCapture(tmp_path)
                        detector = FaceDetector()
                        session = FacialSession()
                        frame_count = 0
                        sample_frames = []
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        step = max(1, total // 30)   # max 30 sample frames

                        prog = st.progress(0)
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame_count % step == 0:
                                _, ef = analyse_frame(frame, detector)
                                if ef:
                                    session.frames.append(ef)
                                    sample_frames.append(frame)
                            frame_count += 1
                            prog.progress(min(frame_count / max(total, 1), 1.0))
                        cap.release()
                        prog.empty()

                        if session.frames:
                            session = compute_session_metrics(session)
                            st.session_state.facial_session = session
                            st.success(f"✓ Analysed {len(session.frames)} frames | "
                                       f"Pattern: **{session.dominant_pattern.upper()}**")
                        else:
                            st.warning("No faces detected in video.")

        else:  # Live webcam
            st.markdown("""
<div class="webcam-container">
<div style="font-size:2rem;margin-bottom:0.5rem">📸</div>
<div style="color:#8b9ab0;font-size:0.88rem">
Use the camera input below to capture a photo.<br>
For real-time streaming, run with: <code style="color:#2dd4bf">streamlit run app.py --server.enableCORS false</code>
</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            cam_img = st.camera_input("📸 Take a photo for analysis", label_visibility="visible")
            if cam_img:
                import cv2
                file_bytes = np.frombuffer(cam_img.getvalue(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    from modules.facial_analysis import (
                        FaceDetector, FacialSession,
                        analyse_frame, compute_session_metrics
                    )
                    detector = FaceDetector()
                    annotated, ef = analyse_frame(img, detector)
                    _, buf = cv2.imencode(".jpg", annotated)
                    st.image(buf.tobytes(), use_container_width=True)

                    if ef:
                        session = FacialSession()
                        for _ in range(8):
                            session.frames.append(ef)
                        session = compute_session_metrics(session)
                        st.session_state.facial_session = session

                        ec1, ec2, ec3 = st.columns(3)
                        with ec1:
                            st.metric("Dominant Emotion", ef.dominant.capitalize())
                        with ec2:
                            st.metric("Valence", f"{ef.valence:+.2f}")
                        with ec3:
                            st.metric("Arousal", f"{ef.arousal:+.2f}")
                    else:
                        st.info("No face detected. Ensure good lighting and face the camera directly.")

    # ── Session metrics display ──────────────────────────────────────────────
    if st.session_state.facial_session:
        fs = st.session_state.facial_session
        st.divider()
        st.markdown("### 🧩 Session Affective Metrics")

        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            c = "#f87171" if fs.mania_score > 55 else "#fbbf24" if fs.mania_score > 30 else "#2dd4bf"
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:{c}">'
                        f'{fs.mania_score:.0f}</div><div class="label">Facial Mania</div></div>',
                        unsafe_allow_html=True)
        with sm2:
            c = "#f87171" if fs.depression_score > 55 else "#fbbf24" if fs.depression_score > 30 else "#60a5fa"
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:{c}">'
                        f'{fs.depression_score:.0f}</div><div class="label">Facial Depression</div></div>',
                        unsafe_allow_html=True)
        with sm3:
            c = "#a78bfa" if fs.mixed_state_score > 40 else "#8b9ab0"
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:{c}">'
                        f'{fs.mixed_state_score:.0f}</div><div class="label">Mixed State</div></div>',
                        unsafe_allow_html=True)
        with sm4:
            c = "#f87171" if fs.affective_instability > 0.5 else "#fbbf24" if fs.affective_instability > 0.2 else "#2dd4bf"
            st.markdown(f'<div class="metric-tile"><div class="value" style="color:{c}">'
                        f'{fs.affective_instability:.2f}</div><div class="label">Instability (MSSD)</div></div>',
                        unsafe_allow_html=True)

        # Valence/arousal timeline
        if len(fs.valence_history) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_va = make_subplots(rows=2, cols=1,
                                   subplot_titles=("Valence Timeline", "Arousal Timeline"),
                                   shared_xaxes=True, vertical_spacing=0.12)
            x = list(range(len(fs.valence_history)))
            fig_va.add_trace(go.Scatter(x=x, y=fs.valence_history, mode="lines",
                                        line=dict(color="#2dd4bf", width=2),
                                        fill="tozeroy", fillcolor="rgba(45,212,191,0.08)"), row=1, col=1)
            fig_va.add_trace(go.Scatter(x=x, y=fs.arousal_history, mode="lines",
                                        line=dict(color="#a78bfa", width=2),
                                        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"), row=2, col=1)
            fig_va.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
                font=dict(family="DM Sans", color="#8b9ab0"),
                showlegend=False, height=280,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            for i in range(1, 3):
                fig_va.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=i, col=1)
                fig_va.update_yaxes(gridcolor="rgba(255,255,255,0.05)", range=[-1.2, 1.2], row=i, col=1)
            st.plotly_chart(fig_va, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · SCREENING RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.text_results and not st.session_state.facial_session:
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem">
  <div style="font-size:3rem;margin-bottom:1rem">📊</div>
  <div style="color:#8b9ab0;font-size:1rem">
    Complete at least one analysis module (text or facial) to see combined screening results.
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        # Compute composite result
        run_composite = st.button("⚡  Compute Composite Screening Result", use_container_width=True)

        if run_composite or st.session_state.screening_result:
            if run_composite:
                from modules.screening_engine import compute_screening_result
                result = compute_screening_result(
                    facial_session=st.session_state.facial_session,
                    text_results=st.session_state.text_results,
                    patient_info={
                        "name": st.session_state.patient_name,
                        "age":  st.session_state.patient_age,
                        "gender": st.session_state.patient_gender,
                    }
                )
                st.session_state.screening_result = result
            else:
                result = st.session_state.screening_result

            # ── Risk Banner ───────────────────────────────────────────────
            RISK_STYLE = {
                "high":     ("rgba(244,63,94,0.2)",  "#f87171",  "rgba(244,63,94,0.4)"),
                "moderate": ("rgba(245,158,11,0.2)", "#fbbf24",  "rgba(245,158,11,0.4)"),
                "low":      ("rgba(45,212,191,0.2)", "#2dd4bf",  "rgba(45,212,191,0.4)"),
                "minimal":  ("rgba(96,165,250,0.15)","#93c5fd",  "rgba(96,165,250,0.3)"),
            }
            bg, fg, br = RISK_STYLE.get(result.overall_risk, RISK_STYLE["minimal"])

            st.markdown(f"""
<div style="background:{bg};border:1px solid {br};border-radius:14px;padding:1.5rem 2rem;
            text-align:center;margin:0.5rem 0 1.5rem">
  <div style="font-size:0.75rem;color:{fg};letter-spacing:0.12em;
              text-transform:uppercase;font-family:'DM Mono';margin-bottom:0.5rem">
    Composite Risk Assessment · Session {result.session_id}
  </div>
  <div style="font-size:2.5rem;font-family:'DM Serif Display';color:{fg};margin:0.2rem 0">
    {result.overall_risk.upper()} RISK
  </div>
  <div style="font-size:1rem;color:{fg};opacity:0.8">
    Dominant Pattern: <b>{result.dominant_state.upper()}</b>
    &nbsp;·&nbsp; Confidence: <b>{result.confidence_pct:.0f}%</b>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Composite score gauge chart ────────────────────────────────
            rc1, rc2 = st.columns(2)

            with rc1:
                fig_gauge = go.Figure()
                for label, val, color, ythick in [
                    ("Mania",      result.composite_mania_score,      "#f87171", 0.6),
                    ("Depression", result.composite_depression_score,  "#60a5fa", 0.45),
                    ("Mixed",      result.composite_mixed_score,       "#a78bfa", 0.30),
                    ("Instability",result.affective_instability_index, "#fbbf24", 0.15),
                ]:
                    fig_gauge.add_trace(go.Bar(
                        x=[val], y=[label], orientation="h",
                        marker_color=color, width=ythick * 2,
                        text=f"{val:.0f}", textposition="outside",
                        textfont=dict(color="#e8edf5"),
                    ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
                    font=dict(family="DM Sans", color="#8b9ab0"),
                    title=dict(text="Composite Scores (0–100)", font=dict(color="#e8edf5", size=13)),
                    xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0)"),
                    showlegend=False, barmode="overlay",
                    margin=dict(l=10, r=30, t=50, b=20), height=280,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with rc2:
                # Donut chart showing modality contribution
                labels, values, colors_pie = [], [], []
                if result.has_facial_data:
                    labels.append("Facial Affect (40%)")
                    values.append(40)
                    colors_pie.append("#2dd4bf")
                if result.has_text_data:
                    labels.append("Linguistic Analysis (60%)")
                    values.append(60)
                    colors_pie.append("#60a5fa")
                if not labels:
                    labels, values = ["No data"], [1]
                    colors_pie = ["#8b9ab0"]

                fig_donut = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.6,
                    marker=dict(colors=colors_pie,
                                line=dict(color=["rgba(0,0,0,0.2)"] * len(labels), width=2)),
                    textfont=dict(color="#e8edf5", size=11),
                    hovertemplate="%{label}<extra></extra>",
                ))
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#8b9ab0"),
                    title=dict(text="Modality Weights", font=dict(color="#e8edf5", size=13)),
                    annotations=[dict(text=f"{result.confidence_pct:.0f}%<br>confidence",
                                      x=0.5, y=0.5, showarrow=False,
                                      font=dict(color="#e8edf5", size=13, family="DM Serif Display"))],
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b9ab0")),
                    margin=dict(l=10, r=10, t=50, b=20), height=280,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # ── Clinical Summary ──────────────────────────────────────────
            import re
            clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", result.clinical_summary)
            st.markdown(f"""
<div class="bpd-card" style="border-color:rgba(45,212,191,0.2)">
<h3>🩺 Clinical Summary</h3>
<p style="color:#b8c4d4;font-size:0.92rem;line-height:1.8;margin:0">{clean}</p>
</div>""", unsafe_allow_html=True)

            # ── Red flags ────────────────────────────────────────────────
            if result.red_flags:
                flags_html = "".join(
                    f'<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);'
                    f'font-size:0.85rem;color:#fca5a5">⚠ {f}</div>'
                    for f in result.red_flags
                )
                st.markdown(f"""
<div class="bpd-card" style="border-color:rgba(244,63,94,0.3)">
<h3 style="color:#f87171 !important">🔴 Critical Flags</h3>
{flags_html}
</div>""", unsafe_allow_html=True)

            # ── Recommendations ───────────────────────────────────────────
            recs_html = ""
            for i, rec in enumerate(result.recommendations, 1):
                icon = "🔴" if "URGENT" in rec or "IMMEDIATE" in rec else "💡" if "resources" in rec.lower() else f"{i}."
                color = "#f87171" if "URGENT" in rec else "#fbbf24" if "crisis" in rec.lower() else "#b8c4d4"
                recs_html += (
                    f'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);'
                    f'font-size:0.87rem;color:{color};line-height:1.6">'
                    f'<b>{icon}</b> {rec}</div>'
                )

            st.markdown(f"""
<div class="bpd-card">
<h3>✅ Recommended Actions</h3>
{recs_html}
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 · REPORT & EXPORT
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="bpd-card"><h3>📄 Export Screening Report</h3>'
                '<p style="color:#8b9ab0;font-size:0.85rem;margin:0">Generate a printable PDF report '
                'suitable for sharing with a mental health provider.</p></div>',
                unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        rpt_name = st.text_input("Patient name for report",
                                  value=st.session_state.patient_name or "Anonymous")
    with rc2:
        rpt_clinician = st.text_input("Clinician name", placeholder="Dr. ...")

    rpt_dob = st.text_input("Date of birth (for report)", placeholder="DD/MM/YYYY or not provided")

    generate_btn = st.button("📥  Generate PDF Report", use_container_width=True)

    if generate_btn:
        if not st.session_state.screening_result:
            st.warning("Please compute a screening result first (go to '📊 Screening Results' tab).")
        else:
            with st.spinner("Generating PDF report..."):
                from modules.report_generator import generate_pdf_report
                pdf_bytes = generate_pdf_report(
                    st.session_state.screening_result,
                    patient_name=rpt_name or "Anonymous",
                    dob=rpt_dob or "Not provided",
                    clinician=rpt_clinician or "Not specified",
                )
                if pdf_bytes:
                    fname = f"bpdisdet_report_{st.session_state.screening_result.session_id}.pdf"
                    st.download_button(
                        label="⬇️  Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✓ PDF generated. Click the button above to download.")
                else:
                    st.error("PDF generation failed. Ensure fpdf2 is installed: pip install fpdf2")

    st.divider()

    # JSON export
    if st.session_state.screening_result:
        result = st.session_state.screening_result
        export_data = {
            "session_id": result.session_id,
            "timestamp": datetime.fromtimestamp(result.timestamp).isoformat(),
            "patient": {
                "name": st.session_state.patient_name or "Anonymous",
                "age": st.session_state.patient_age,
                "gender": st.session_state.patient_gender,
            },
            "scores": {
                "mania_composite": round(result.composite_mania_score, 2),
                "depression_composite": round(result.composite_depression_score, 2),
                "mixed_composite": round(result.composite_mixed_score, 2),
                "affective_instability": round(result.affective_instability_index, 2),
            },
            "classification": {
                "dominant_state": result.dominant_state,
                "overall_risk": result.overall_risk,
                "confidence_pct": round(result.confidence_pct, 1),
            },
            "clinical_summary": result.clinical_summary,
            "recommendations": result.recommendations,
            "red_flags": result.red_flags,
            "modalities": {
                "facial": result.has_facial_data,
                "text": result.has_text_data,
            }
        }

        st.download_button(
            "⬇️  Export Results as JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"bpdisdet_{result.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.markdown("""
<div style="font-size:0.8rem;color:#6b7280;text-align:center;line-height:2.2;padding:1rem">
  <b style="color:#8b9ab0">bpdisdet v1.0</b> · Multimodal Mental Health Screening<br>
  Supporting UN SDG-3: Good Health and Well-Being<br>
  Built with OpenCV · DeepFace · Anthropic Claude · Streamlit<br>
  <span style="color:#4ade80">For research and early screening use only. Not a clinical diagnostic instrument.</span><br>
  Crisis support: <b style="color:#e8edf5">iCall 9152987821</b> · <b style="color:#e8edf5">Vandrevala 1860-2662-345</b>
</div>
""", unsafe_allow_html=True)