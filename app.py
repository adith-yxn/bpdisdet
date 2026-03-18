"""
app.py  ·  bpdisdet v2
══════════════════════════════════════════════════════════════════════════
Multimodal Mental Health Screening Tool — Bipolar Spectrum Edition
Three modalities: Facial Affect · Linguistic Analysis · Validated Questionnaires
SDG-3: Good Health & Well-Being · Equitable Early Diagnostics
"""

import os, sys, json, time, base64, tempfile
from io import BytesIO
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="bpdisdet v2 · Mental Health Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg0:#080d14; --bg1:#0e1520; --bg2:#151e2d; --bg3:#1c2840;
  --teal:#2dd4bf; --teal-dim:rgba(45,212,191,.15);
  --amber:#f59e0b; --rose:#f43f5e; --violet:#8b5cf6; --blue:#3b82f6;
  --fg:#dde5f0; --fg2:#8fa0b8; --fg3:#4a5a72;
  --border:rgba(255,255,255,.06); --glow:0 0 40px rgba(45,212,191,.07);
  --r:10px;
}
html,body,[class*="css"],.stApp{background:var(--bg0)!important;color:var(--fg)!important;font-family:'Inter',sans-serif!important}
.stApp{background:var(--bg0)}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--teal),var(--blue),var(--violet));z-index:9999}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{font-family:'Inter',sans-serif!important}

/* Cards */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);
  padding:1.3rem 1.5rem;margin-bottom:.9rem;transition:border-color .2s}
.card:hover{border-color:rgba(45,212,191,.18)}
.card h3{font-family:'Syne',sans-serif!important;font-size:1.05rem!important;
  color:var(--teal)!important;margin:0 0 .5rem!important}

/* Hero */
.hero{text-align:center;padding:2rem 1rem 1.2rem}
.hero h1{font-family:'Syne',sans-serif!important;font-size:clamp(1.8rem,4vw,2.8rem)!important;
  font-weight:800!important;color:var(--teal)!important;letter-spacing:-.03em;margin:0!important}
.hero .sub{color:var(--fg2);font-size:.9rem;margin-top:.3rem;letter-spacing:.06em;text-transform:uppercase}
.badge{display:inline-block;background:rgba(74,222,128,.12);border:1px solid rgba(74,222,128,.25);
  color:#4ade80;font-size:.7rem;padding:3px 11px;border-radius:20px;
  letter-spacing:.09em;font-family:'JetBrains Mono',monospace;margin-top:.5rem}

/* Metric tiles */
.tile{background:var(--bg3);border:1px solid var(--border);border-radius:8px;
  padding:.9rem;text-align:center}
.tile .val{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:500;line-height:1.1}
.tile .lbl{font-size:.72rem;color:var(--fg2);margin-top:3px;text-transform:uppercase;letter-spacing:.06em}

/* Risk chips */
.chip{display:inline-block;padding:4px 13px;border-radius:20px;
  font-family:'JetBrains Mono',monospace;font-size:.76rem;letter-spacing:.05em}
.chip-high    {background:rgba(244,63,94,.18);color:#f87171;border:1px solid rgba(244,63,94,.3)}
.chip-moderate{background:rgba(245,158,11,.18);color:#fbbf24;border:1px solid rgba(245,158,11,.3)}
.chip-low     {background:rgba(45,212,191,.14);color:#2dd4bf;border:1px solid rgba(45,212,191,.25)}
.chip-minimal {background:rgba(96,165,250,.13);color:#93c5fd;border:1px solid rgba(96,165,250,.25)}

/* Inputs */
.stTextArea textarea,.stTextInput input{
  background:var(--bg3)!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--fg)!important;font-family:'Inter',sans-serif!important}
.stTextArea textarea:focus,.stTextInput input:focus{
  border-color:rgba(45,212,191,.4)!important;box-shadow:0 0 0 2px rgba(45,212,191,.08)!important}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,var(--teal),#0891b2)!important;
  color:#080d14!important;border:none!important;border-radius:8px!important;
  font-family:'Syne',sans-serif!important;font-weight:600!important;font-size:.86rem!important;
  padding:.5rem 1.3rem!important;transition:all .2s!important;letter-spacing:.02em}
.stButton>button:hover{transform:translateY(-1px)!important;
  box-shadow:0 4px 20px rgba(45,212,191,.28)!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--bg1)!important;border-radius:var(--r);
  padding:4px;border:1px solid var(--border);gap:2px}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--fg2)!important;
  border-radius:7px!important;font-family:'Inter',sans-serif!important;
  font-size:.86rem!important;padding:8px 16px!important;border:none!important}
.stTabs [aria-selected="true"]{background:var(--bg2)!important;color:var(--teal)!important;
  border:1px solid rgba(45,212,191,.2)!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem!important}

/* Selectbox / Radio */
.stSelectbox>div>div,.stRadio>div{background:var(--bg3)!important;border-radius:8px!important}

/* Progress */
.stProgress>div>div{background:linear-gradient(90deg,var(--teal),var(--blue))!important;border-radius:4px!important}

/* Disclaimer */
.disclaimer{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
  border-radius:var(--r);padding:.75rem 1.1rem;font-size:.82rem;color:#fbbf24;line-height:1.6;
  margin-bottom:1.1rem}

/* Q items */
.q-item{background:var(--bg3);border:1px solid var(--border);border-radius:8px;
  padding:.9rem 1.1rem;margin-bottom:.5rem}
.q-item .q-text{font-size:.9rem;color:var(--fg);line-height:1.5;margin-bottom:.5rem}
.q-item .q-cat{font-size:.7rem;color:var(--fg3);font-family:'JetBrains Mono',monospace;
  text-transform:uppercase;letter-spacing:.05em}

/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(45,212,191,.25)}

#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "facial_session":       None,
    "text_results":         [],
    "questionnaire_result": None,
    "screening_result":     None,
    "api_key":              os.environ.get("ANTHROPIC_API_KEY", ""),
    "patient_name":         "",
    "patient_age":          25,
    "patient_gender":       "Not specified",
    "demo_text":            "",
    "mdq_answers":          {},
    "phq9_answers":         {},
    "als_answers":          {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _col(score):
    """Return hex colour for a 0–100 score."""
    if score >= 65: return "#f87171"
    if score >= 40: return "#fbbf24"
    if score >= 20: return "#2dd4bf"
    return "#93c5fd"

def _chip(risk):
    return f'<span class="chip chip-{risk}">{risk.upper()}</span>'

def _tile(value, label, color="#2dd4bf"):
    return (f'<div class="tile"><div class="val" style="color:{color}">{value}</div>'
            f'<div class="lbl">{label}</div></div>')

def _plotly_defaults():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,30,45,.8)",
                font=dict(family="Inter", color="#8fa0b8"),
                margin=dict(l=10, r=10, t=45, b=15))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<h2 style="font-family:Syne,sans-serif;color:#2dd4bf;margin:0">🧠 bpdisdet</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5a72;font-size:.8rem;margin:2px 0 12px">v2 · Bipolar Spectrum Screening</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**⚙️ API Configuration**")
    api_in = st.text_input("Anthropic API Key", value=st.session_state.api_key,
                            type="password", placeholder="sk-ant-...")
    if api_in:
        st.session_state.api_key = api_in
        st.success("✓ API key set — enhanced LLM analysis enabled", icon="🔐")
    else:
        st.info("Running in offline mode (heuristic analyser)", icon="💡")

    st.divider()
    st.markdown("**👤 Patient Details**")
    st.session_state.patient_name   = st.text_input("Name / Alias", value=st.session_state.patient_name, placeholder="Anonymous")
    st.session_state.patient_age    = st.slider("Age", 12, 90, st.session_state.patient_age)
    st.session_state.patient_gender = st.selectbox("Gender", ["Not specified","Female","Male","Non-binary","Prefer not to say"])
    clinician_in = st.text_input("Clinician", placeholder="Dr. ...")
    st.divider()

    # Progress indicator
    n_done = sum([
        st.session_state.facial_session is not None,
        len(st.session_state.text_results) > 0,
        st.session_state.questionnaire_result is not None,
    ])
    st.markdown(f"**📋 Modules completed: {n_done}/3**")
    st.progress(n_done / 3)
    st.markdown(f"{'✅' if st.session_state.facial_session else '⬜'} Facial analysis")
    st.markdown(f"{'✅' if st.session_state.text_results else '⬜'} Text analysis")
    st.markdown(f"{'✅' if st.session_state.questionnaire_result else '⬜'} Questionnaires")
    st.divider()

    st.markdown('<div style="font-size:.75rem;color:#4a5a72;text-align:center;line-height:1.8">'
                'v2.0 · SDG-3 Aligned<br>Research & Screening Only<br>Not a diagnostic instrument'
                '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>bpdisdet</h1>
  <div class="sub">Multimodal Bipolar Spectrum · Early Screening System · v2</div>
  <div class="badge">✦ UN SDG-3 · 3-Modality Analysis · No TensorFlow · Cloud Ready</div>
</div>
<div class="disclaimer">
  ⚠ <b>Medical Disclaimer:</b> This tool provides <i>screening support only</i> — not a clinical diagnosis.
  All results must be reviewed by a licensed mental health professional.
  Crisis support: <b>iCall (India): 9152987821</b> · <b>Vandrevala: 1860-2662-345</b> · <b>Emergency: 112</b>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Text Analysis",
    "📷 Facial Affect",
    "📋 Questionnaires",
    "📊 Results",
    "📄 Report",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1  ·  TEXT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="card"><h3>🔤 Psycholinguistic Marker Analysis</h3>'
                '<p style="color:#8fa0b8;font-size:.84rem;margin:0">Submit free-text entries. '
                'Claude API provides DSM-5-TR aligned analysis; local heuristic runs offline.</p></div>',
                unsafe_allow_html=True)

    col_in, col_guide = st.columns([3, 1])

    with col_guide:
        st.markdown("""<div class="card"><h3>💡 Prompts</h3>
<div style="font-size:.8rem;color:#8fa0b8;line-height:1.9">
<b style="color:#dde5f0">Describe:</b><br>
• How you've felt this week<br>
• Energy &amp; sleep patterns<br>
• Current plans &amp; projects<br>
• Things you enjoy / avoid<br>
• Racing or slowed thoughts<br>
• Mood swings or irritability<br><br>
<b style="color:#4ade80">50–300 words optimal</b>
</div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card"><h3>🔍 Markers</h3>
<div style="font-size:.77rem;line-height:1.9">
<b style="color:#f87171">Mania:</b> Pressured speech · Grandiosity · Flight of ideas · Decreased sleep<br>
<b style="color:#60a5fa">Depression:</b> Anhedonia · Hopelessness · Psychomotor slowing · Worthlessness<br>
<b style="color:#a78bfa">Mixed:</b> Dysphoric arousal · Irritability · Racing negative thoughts
</div></div>""", unsafe_allow_html=True)

    with col_in:
        mode = st.radio("Mode", ["Single entry", "Multi-entry journal (3 days)"],
                        horizontal=True, label_visibility="collapsed")
        entries = []

        if mode == "Single entry":
            txt = st.text_area("Entry", height=190,
                               placeholder="Describe how you've been feeling lately — be as open as you like...",
                               label_visibility="collapsed")
            if txt.strip():
                entries = [txt]
        else:
            st.caption("One entry per day — leave empty if unavailable")
            for d in range(1, 4):
                e = st.text_area(f"Day {d}",  height=110,
                                 placeholder=f"How were you feeling {['today','yesterday','2 days ago'][d-1]}?",
                                 key=f"jday{d}", label_visibility="visible")
                if e.strip(): entries.append(e)

        # Demo buttons
        d1, d2, d3 = st.columns(3)
        with d1:
            if st.button("📋 Demo: Manic", use_container_width=True):
                st.session_state.demo_text = (
                    "I have SO many ideas right now I can barely sleep — honestly I don't even NEED sleep! "
                    "I feel more powerful and alive than I ever have. Started three new companies this week, "
                    "already talking to investors. People literally can't keep up with my mind. "
                    "I was DESTINED for this. God or the universe or whatever is sending me messages. "
                    "Everything is clicking into place at lightning speed!!! I am unstoppable right now."
                )
        with d2:
            if st.button("📋 Demo: Depressed", use_container_width=True):
                st.session_state.demo_text = (
                    "I don't really see the point anymore. Everything feels heavy and grey. "
                    "Haven't left the house in days... maybe a week. Can't remember. "
                    "I used to love painting but the brushes just sit there. I feel worthless. "
                    "Like a burden to everyone. Thoughts are so slow, like wading through mud. "
                    "Nothing will get better. I just feel empty and exhausted all the time."
                )
        with d3:
            if st.button("📋 Demo: Mixed", use_container_width=True):
                st.session_state.demo_text = (
                    "My mind won't stop racing but everything I think about fills me with dread. "
                    "I'm furious at everyone for no reason then crying an hour later. "
                    "I have this horrible energy — restless, agitated — but I hate it. "
                    "I feel like I need to do something but every idea feels pointless. "
                    "Can't sleep but I'm exhausted. Started 4 projects, abandoned all of them the same day."
                )

        if st.session_state.demo_text and not entries:
            st.info("Demo text loaded — click Analyse below")
            entries = [st.session_state.demo_text]
            with st.expander("View loaded demo text"):
                st.write(st.session_state.demo_text)

        if st.button("🔍  Analyse Text", use_container_width=True):
            if not entries:
                st.warning("Enter at least one text entry first.")
            else:
                with st.spinner("Running psycholinguistic analysis..."):
                    from modules.text_analysis import analyse_with_api, analyse_heuristic
                    results, prog = [], st.progress(0)
                    for i, entry in enumerate(entries):
                        if st.session_state.api_key:
                            results.append(analyse_with_api(entry, st.session_state.api_key))
                        else:
                            results.append(analyse_heuristic(entry))
                        prog.progress((i+1)/len(entries))
                    st.session_state.text_results = results
                    prog.empty()
                    st.session_state.demo_text = ""
                    method = results[0].analysis_method
                    st.success(f"✓ Analysed {len(results)} entr{'y' if len(results)==1 else 'ies'} · Method: **{method}**")

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.text_results:
        res = st.session_state.text_results
        latest = res[-1]
        st.divider()
        st.markdown("### Analysis Results")

        avg_m = sum(r.mania_score    for r in res) / len(res)
        avg_d = sum(r.depression_score for r in res) / len(res)
        avg_x = sum(r.mixed_score    for r in res) / len(res)

        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1: st.markdown(_tile(f"{avg_m:.0f}", "Mania Index", _col(avg_m)), unsafe_allow_html=True)
        with tc2: st.markdown(_tile(f"{avg_d:.0f}", "Depression Index", _col(avg_d)), unsafe_allow_html=True)
        with tc3: st.markdown(_tile(f"{avg_x:.0f}", "Mixed State", _col(avg_x)), unsafe_allow_html=True)
        with tc4: st.markdown(_tile(_chip(latest.risk_level), "Risk Level"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            m = latest.markers
            cats = ["Pressured Speech","Flight of Ideas","Grandiosity",
                    "Anhedonia","Hopelessness","Worthlessness",
                    "Irritability","Mixed Dysphoria","Somatic"]
            vals = [m.pressured_speech, m.flight_of_ideas, m.grandiosity,
                    m.anhedonia, m.hopelessness, m.worthlessness,
                    m.irritability, m.mixed_dysphoria, m.somatic_complaints]
            fig = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", fillcolor="rgba(45,212,191,.1)",
                line=dict(color="#2dd4bf", width=2),
                marker=dict(color="#2dd4bf", size=5)))
            fig.update_layout(**_plotly_defaults(),
                polar=dict(bgcolor="rgba(21,30,45,.8)",
                           radialaxis=dict(visible=True, range=[0,100],
                                          gridcolor="rgba(255,255,255,.05)",
                                          tickfont=dict(color="#4a5a72",size=8)),
                           angularaxis=dict(gridcolor="rgba(255,255,255,.05)",
                                            tickfont=dict(color="#8fa0b8",size=9))),
                title=dict(text="Linguistic Marker Profile", font=dict(color="#dde5f0",size=12)),
                height=310)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            if len(res) > 1:
                xl = [f"Entry {i+1}" for i in range(len(res))]
                fig2 = go.Figure()
                for name, col, y in [
                    ("Mania","#f87171",[r.mania_score for r in res]),
                    ("Depression","#60a5fa",[r.depression_score for r in res]),
                    ("Mixed","#a78bfa",[r.mixed_score for r in res]),
                ]:
                    fig2.add_trace(go.Scatter(x=xl, y=y, name=name,
                        line=dict(color=col,width=2), mode="lines+markers", marker=dict(size=7)))
                fig2.update_layout(**_plotly_defaults(),
                    title=dict(text="Longitudinal Trend", font=dict(color="#dde5f0",size=12)),
                    yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,.03)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"), height=310)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Stylometric features
                m = latest.markers
                style_names  = ["Words/Sentence","Lexical Diversity","Exclamation Density",
                                 "Caps Ratio","Positive Sentiment","Negative Sentiment"]
                style_vals   = [min(100, m.words_per_sentence*3), m.lexical_diversity,
                                 min(100, m.exclamation_density*10), min(100, m.caps_ratio*300),
                                 m.sentiment_positive, m.sentiment_negative]
                fig2 = go.Figure(go.Bar(
                    x=style_names, y=style_vals,
                    marker_color=["#2dd4bf","#60a5fa","#f87171","#fbbf24","#4ade80","#f87171"],
                    text=[f"{v:.0f}" for v in style_vals], textposition="outside",
                    textfont=dict(color="#dde5f0")))
                fig2.update_layout(**_plotly_defaults(),
                    title=dict(text="Stylometric Features", font=dict(color="#dde5f0",size=12)),
                    yaxis=dict(range=[0,120], gridcolor="rgba(255,255,255,.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0)"),
                    height=310)
                st.plotly_chart(fig2, use_container_width=True)

        if latest.clinical_summary:
            import re
            clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", latest.clinical_summary)
            st.markdown(f'<div class="card"><h3>🩺 Clinical Summary</h3>'
                        f'<p style="color:#b8c8de;font-size:.9rem;line-height:1.75;margin:0">{clean}</p></div>',
                        unsafe_allow_html=True)

        if latest.key_phrases:
            ph_html = " ".join(
                f'<span style="background:rgba(244,63,94,.14);color:#f87171;'
                f'border:1px solid rgba(244,63,94,.3);padding:3px 9px;border-radius:20px;'
                f'font-size:.77rem;margin:2px;display:inline-block">{p}</span>'
                if p.startswith("⚠") else
                f'<span style="background:rgba(45,212,191,.1);color:#2dd4bf;'
                f'border:1px solid rgba(45,212,191,.22);padding:3px 9px;border-radius:20px;'
                f'font-size:.77rem;margin:2px;display:inline-block">{p}</span>'
                for p in latest.key_phrases)
            st.markdown(f'<div class="card"><h3>🏷️ Key Markers</h3>{ph_html}</div>',
                        unsafe_allow_html=True)

        if latest.suicidal_flag:
            st.error("🔴 URGENT: Suicidal ideation markers detected in text. "
                     "Please contact a crisis helpline immediately. iCall: 9152987821")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2  ·  FACIAL AFFECT
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><h3>📷 Geometric Facial Affect Analysis</h3>'
                '<p style="color:#8fa0b8;font-size:.84rem;margin:0">Pure OpenCV geometric pipeline — '
                'no TensorFlow, no DeepFace. Works on Streamlit Cloud.</p></div>',
                unsafe_allow_html=True)

    fc1, fc2 = st.columns([3, 2])

    with fc2:
        st.markdown("""<div class="card"><h3>📐 Features Extracted</h3>
<div style="font-size:.8rem;color:#8fa0b8;line-height:2">
<b style="color:#2dd4bf">EAR</b> — Eye Aspect Ratio<br>
Blink rate, alertness, psychomotor<br><br>
<b style="color:#60a5fa">MAR</b> — Mouth Aspect Ratio<br>
Expressiveness, speech activity<br><br>
<b style="color:#a78bfa">BDR</b> — Brow Displacement<br>
Worry, surprise, anger<br><br>
<b style="color:#f87171">FAI</b> — Facial Asymmetry<br>
Instability proxy<br><br>
<b style="color:#fbbf24">SLV</b> — Skin Luminance Var<br>
Arousal / emotional activation
</div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card"><h3>🎭 Emotion Labels</h3>
<div style="font-size:.8rem;line-height:2.1">
<span style="color:#4ade80">● Joyful</span> — elevated valence<br>
<span style="color:#fbbf24">● Elevated</span> — high arousal<br>
<span style="color:#f87171">● Agitated</span> — manic/mixed<br>
<span style="color:#a78bfa">● Anxious</span> — mixed/fear<br>
<span style="color:#60a5fa">● Withdrawn</span> — low valence<br>
<span style="color:#8fa0b8">● Flat</span> — depressed<br>
<span style="color:#ef4444">● Distressed</span> — depressive<br>
<span style="color:#64748b">● Neutral</span> — baseline
</div></div>""", unsafe_allow_html=True)

    with fc1:
        mode_f = st.radio("Capture mode", ["📁 Upload photo/video", "📸 Webcam capture"],
                          horizontal=True, label_visibility="collapsed")

        def _run_facial_on_image(img_bgr):
            import cv2
            from modules.facial_analysis import (CascadeDetector, FacialSession,
                                                  analyse_frame, compute_session_metrics)
            detector = CascadeDetector()
            annotated, ef = analyse_frame(img_bgr, detector)
            _, buf = cv2.imencode(".jpg", annotated)
            st.image(buf.tobytes(), use_container_width=True)
            if ef:
                session = FacialSession()
                # Simulate a brief session by duplicating the single frame
                # with slight noise to get meaningful MSSD
                import copy
                for jitter in np.linspace(-0.05, 0.05, 10):
                    frame_copy = copy.deepcopy(ef)
                    frame_copy.valence = max(-1, min(1, ef.valence + jitter))
                    session.frames.append(frame_copy)
                session = compute_session_metrics(session)
                st.session_state.facial_session = session

                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1: st.metric("Emotion",  ef.emotion.capitalize())
                with ec2: st.metric("Valence",  f"{ef.valence:+.2f}")
                with ec3: st.metric("Arousal",  f"{ef.arousal:+.2f}")
                with ec4: st.metric("Conf.",    f"{ef.confidence*100:.0f}%")

                # Feature scores
                st.markdown("**Geometric feature scores:**")
                feats = ef.features
                for lbl, val, scale in [
                    ("Eye AR (EAR)",   feats.ear,  0.5),
                    ("Mouth AR (MAR)", feats.mar,  0.4),
                    ("Brow Disp (BDR)",feats.bdr,  0.6),
                    ("Face Asym (FAI)",feats.fai,  0.2),
                ]:
                    pct = int(min(100, val/scale*100))
                    st.progress(pct, text=f"{lbl}: {val:.3f} → {pct}%")
                return True
            else:
                st.info("No face detected — ensure good lighting and face the camera directly.")
                return False

        if "📁" in mode_f:
            uploaded = st.file_uploader("Upload photo or video",
                                        type=["jpg","jpeg","png","mp4","avi","mov"],
                                        label_visibility="collapsed")
            if uploaded:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                import cv2
                if uploaded.type.startswith("image"):
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        _run_facial_on_image(img)
                    else:
                        st.error("Could not decode image file.")
                else:  # video
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(file_bytes.tobytes())
                        tmp_path = tmp.name
                    with st.spinner("Analysing video frames..."):
                        from modules.facial_analysis import (CascadeDetector, FacialSession,
                                                              analyse_frame, compute_session_metrics)
                        cap = cv2.VideoCapture(tmp_path)
                        detector = CascadeDetector()
                        session  = FacialSession()
                        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        step     = max(1, total // 40)
                        prog_v   = st.progress(0)
                        fc_i     = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            if fc_i % step == 0:
                                _, ef = analyse_frame(frame, detector)
                                if ef: session.frames.append(ef)
                            prog_v.progress(min(fc_i / max(total,1), 1.0))
                            fc_i += 1
                        cap.release(); prog_v.empty()
                    if session.frames:
                        session = compute_session_metrics(session)
                        st.session_state.facial_session = session
                        st.success(f"✓ {len(session.frames)} frames analysed · Pattern: **{session.dominant_pattern.upper()}**")
                    else:
                        st.warning("No faces detected in video.")
        else:
            cam_img = st.camera_input("Take photo", label_visibility="collapsed")
            if cam_img:
                import cv2
                fb = np.frombuffer(cam_img.getvalue(), np.uint8)
                img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                if img is not None:
                    _run_facial_on_image(img)

    # Session metrics
    if st.session_state.facial_session:
        fs = st.session_state.facial_session
        st.divider()
        st.markdown("### Session Metrics")
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1: st.markdown(_tile(f"{fs.mania_score:.0f}",      "Facial Mania",   _col(fs.mania_score)),      unsafe_allow_html=True)
        with sm2: st.markdown(_tile(f"{fs.depression_score:.0f}", "Facial Depression", _col(fs.depression_score)), unsafe_allow_html=True)
        with sm3: st.markdown(_tile(f"{fs.mixed_state_score:.0f}","Mixed State",     _col(fs.mixed_state_score)), unsafe_allow_html=True)
        with sm4: st.markdown(_tile(f"{fs.affective_instability:.3f}", "MSSD Instability",
                                    "#f87171" if fs.affective_instability > .05 else "#2dd4bf"),
                              unsafe_allow_html=True)

        if len(fs.valence_history) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_va = make_subplots(rows=2, cols=1, subplot_titles=("Valence Timeline","Arousal Timeline"),
                                   shared_xaxes=True, vertical_spacing=.14)
            x = list(range(len(fs.valence_history)))
            fig_va.add_trace(go.Scatter(x=x, y=fs.valence_history, mode="lines",
                line=dict(color="#2dd4bf",width=2), fill="tozeroy", fillcolor="rgba(45,212,191,.07)"), row=1, col=1)
            fig_va.add_trace(go.Scatter(x=x, y=fs.arousal_history, mode="lines",
                line=dict(color="#a78bfa",width=2), fill="tozeroy", fillcolor="rgba(167,139,250,.07)"), row=2, col=1)
            fig_va.update_layout(**_plotly_defaults(), showlegend=False, height=270)
            for i in range(1,3):
                fig_va.update_xaxes(gridcolor="rgba(255,255,255,.04)", row=i, col=1)
                fig_va.update_yaxes(gridcolor="rgba(255,255,255,.04)", range=[-1.3,1.3], row=i, col=1)
            st.plotly_chart(fig_va, use_container_width=True)

        if fs.feature_summary:
            with st.expander("📐 Full feature summary"):
                st.json(fs.feature_summary)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3  ·  QUESTIONNAIRES
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card"><h3>📋 Validated Clinical Questionnaires</h3>'
                '<p style="color:#8fa0b8;font-size:.84rem;margin:0">'
                'MDQ-7 (mania screening) · PHQ-9 (depression severity) · '
                'ALS-SF (affective lability). Rate each item for the past 2 weeks.</p></div>',
                unsafe_allow_html=True)

    from modules.questionnaire import (
        MDQ_QUESTIONS,  MDQ_RESPONSE_OPTIONS,
        PHQ9_QUESTIONS, PHQ9_RESPONSE_OPTIONS,
        ALS_QUESTIONS,  ALS_RESPONSE_OPTIONS,
        score_questionnaire,
    )

    q_tab1, q_tab2, q_tab3 = st.tabs(["📊 MDQ-7 · Mania Screen", "😔 PHQ-9 · Depression", "🌊 ALS-SF · Lability"])

    with q_tab1:
        st.markdown("**Rate each item for the past 2 weeks:**")
        for q in MDQ_QUESTIONS:
            st.markdown(f'<div class="q-item"><div class="q-cat">Marker: {q["category"]}</div>'
                        f'<div class="q-text">{q["text"]}</div></div>', unsafe_allow_html=True)
            ans = st.select_slider(f" ", options=MDQ_RESPONSE_OPTIONS,
                                   key=f"mdq_{q['id']}", label_visibility="collapsed")
            st.session_state.mdq_answers[q["id"]] = MDQ_RESPONSE_OPTIONS.index(ans)

    with q_tab2:
        st.markdown("**Over the last 2 weeks, how often have you been bothered by:**")
        for q in PHQ9_QUESTIONS:
            is_safety = q.get("is_safety_item", False)
            border = "border-color:rgba(244,63,94,.3)" if is_safety else ""
            st.markdown(f'<div class="q-item" style="{border}"><div class="q-cat">'
                        f'{"⚠️ Safety item · " if is_safety else ""}PHQ: {q["category"]}</div>'
                        f'<div class="q-text">{q["text"]}</div></div>', unsafe_allow_html=True)
            ans = st.select_slider(f"  ", options=PHQ9_RESPONSE_OPTIONS,
                                   key=f"phq9_{q['id']}", label_visibility="collapsed")
            st.session_state.phq9_answers[q["id"]] = PHQ9_RESPONSE_OPTIONS.index(ans)

    with q_tab3:
        st.markdown("**Rate how frequently each statement describes you:**")
        for q in ALS_QUESTIONS:
            st.markdown(f'<div class="q-item"><div class="q-cat">Lability: {q["category"]}</div>'
                        f'<div class="q-text">{q["text"]}</div></div>', unsafe_allow_html=True)
            ans = st.select_slider(f"   ", options=ALS_RESPONSE_OPTIONS,
                                   key=f"als_{q['id']}", label_visibility="collapsed")
            st.session_state.als_answers[q["id"]] = ALS_RESPONSE_OPTIONS.index(ans)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊  Score Questionnaires", use_container_width=True):
        if (len(st.session_state.mdq_answers) < len(MDQ_QUESTIONS) or
                len(st.session_state.phq9_answers) < len(PHQ9_QUESTIONS) or
                len(st.session_state.als_answers) < len(ALS_QUESTIONS)):
            st.warning("Please answer all questions in all three scales before scoring.")
        else:
            with st.spinner("Computing questionnaire scores..."):
                qr = score_questionnaire(
                    st.session_state.mdq_answers,
                    st.session_state.phq9_answers,
                    st.session_state.als_answers,
                )
                st.session_state.questionnaire_result = qr
            st.success("✓ Questionnaires scored.")

    if st.session_state.questionnaire_result:
        qr = st.session_state.questionnaire_result
        st.divider()
        st.markdown("### Questionnaire Results")

        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1: st.markdown(_tile(f"{qr.mdq_raw_score}/28", f"MDQ-7 · {qr.mdq_severity.upper()}", _col(qr.mdq_scaled)), unsafe_allow_html=True)
        with qc2: st.markdown(_tile(f"{qr.phq9_raw_score}/27", f"PHQ-9 · {qr.phq9_severity.upper()}", _col(qr.phq9_scaled)), unsafe_allow_html=True)
        with qc3: st.markdown(_tile(f"{qr.als_raw_score}/18", f"ALS-SF · {qr.als_severity.upper()}", _col(qr.als_scaled)), unsafe_allow_html=True)
        with qc4: st.markdown(_tile(f"{qr.composite_score:.0f}", "Composite Score", _col(qr.composite_score)), unsafe_allow_html=True)

        if qr.phq9_safety_flag:
            st.error("🔴 PHQ-9 Item 9 (Suicidal Ideation) was endorsed. "
                     "Please contact a crisis professional immediately. iCall: 9152987821")

        st.markdown("<br>", unsafe_allow_html=True)
        # Category breakdown chart
        cats = list(qr.category_breakdown.keys())
        vals = list(qr.category_breakdown.values())
        colors_bar = ["#f87171" if "MDQ" in c else "#60a5fa" if "PHQ" in c else "#a78bfa"
                      for c in cats]
        fig_q = go.Figure(go.Bar(
            x=[c.split(": ")[-1].replace("_"," ") for c in cats],
            y=vals, marker_color=colors_bar,
            text=[f"{v:.0f}" for v in vals], textposition="outside",
            textfont=dict(color="#dde5f0", size=9)))
        fig_q.update_layout(**_plotly_defaults(),
            title=dict(text="Category Scores (MDQ=red · PHQ=blue · ALS=purple)",
                       font=dict(color="#dde5f0", size=12)),
            yaxis=dict(range=[0,115], gridcolor="rgba(255,255,255,.04)"),
            xaxis=dict(tickangle=-40, gridcolor="rgba(255,255,255,0)"),
            height=320)
        st.plotly_chart(fig_q, use_container_width=True)

        import re
        clean_interp = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", qr.interpretation)
        st.markdown(f'<div class="card"><h3>🩺 Questionnaire Interpretation</h3>'
                    f'<p style="color:#b8c8de;font-size:.9rem;line-height:1.75;margin:0">{clean_interp}</p></div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4  ·  COMPOSITE RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    n_complete = sum([
        st.session_state.facial_session is not None,
        len(st.session_state.text_results) > 0,
        st.session_state.questionnaire_result is not None,
    ])

    if n_complete == 0:
        st.markdown("""<div style="text-align:center;padding:4rem 2rem">
  <div style="font-size:3rem;margin-bottom:1rem">📊</div>
  <div style="color:#8fa0b8;font-size:1rem">
    Complete at least one module to see composite results.<br>
    All three modalities give the most accurate screening.
  </div></div>""", unsafe_allow_html=True)
    else:
        if st.button("⚡  Compute Composite Screening Result", use_container_width=True):
            from modules.screening_engine import compute_screening_result
            result = compute_screening_result(
                facial_session=st.session_state.facial_session,
                text_results=st.session_state.text_results,
                questionnaire_result=st.session_state.questionnaire_result,
                patient_info={"name": st.session_state.patient_name,
                              "age":  st.session_state.patient_age}
            )
            st.session_state.screening_result = result

        if st.session_state.screening_result:
            result = st.session_state.screening_result

            RISK_STYLES = {
                "high":     ("rgba(244,63,94,.2)",  "#f87171", "rgba(244,63,94,.4)"),
                "moderate": ("rgba(245,158,11,.2)", "#fbbf24", "rgba(245,158,11,.4)"),
                "low":      ("rgba(45,212,191,.18)","#2dd4bf", "rgba(45,212,191,.35)"),
                "minimal":  ("rgba(96,165,250,.15)","#93c5fd", "rgba(96,165,250,.3)"),
            }
            bg, fg, br = RISK_STYLES.get(result.overall_risk, RISK_STYLES["minimal"])

            st.markdown(f"""
<div style="background:{bg};border:1px solid {br};border-radius:14px;padding:1.5rem 2rem;
            text-align:center;margin:.5rem 0 1.5rem">
  <div style="font-size:.72rem;color:{fg};letter-spacing:.12em;
              text-transform:uppercase;font-family:'JetBrains Mono',monospace;margin-bottom:.5rem">
    Composite Risk · Session {result.session_id} · {sum([result.has_facial,result.has_text,result.has_questionnaire])}/3 Modalities
  </div>
  <div style="font-size:2.4rem;font-family:'Syne',sans-serif;font-weight:800;color:{fg};margin:.2rem 0">
    {result.overall_risk.upper()} RISK
  </div>
  <div style="font-size:1rem;color:{fg};opacity:.85">
    Pattern: <b>{result.dominant_state.upper()}</b> &nbsp;·&nbsp;
    Confidence: <b>{result.confidence_pct:.0f}%</b>
  </div>
</div>""", unsafe_allow_html=True)

            # Scores row
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1: st.markdown(_tile(f"{result.composite_mania:.0f}",       "Composite Mania",      _col(result.composite_mania)),      unsafe_allow_html=True)
            with rc2: st.markdown(_tile(f"{result.composite_depression:.0f}",  "Composite Depression", _col(result.composite_depression)), unsafe_allow_html=True)
            with rc3: st.markdown(_tile(f"{result.composite_mixed:.0f}",       "Composite Mixed",      _col(result.composite_mixed)),      unsafe_allow_html=True)
            with rc4: st.markdown(_tile(f"{result.affective_instability:.0f}", "Instability Index",    _col(result.affective_instability)),unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)

            with ch1:
                # 9-axis radar across all modalities
                ms = result.modality_scores
                cats_r = list(ms.keys())
                vals_r = list(ms.values())
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals_r+[vals_r[0]], theta=cats_r+[cats_r[0]],
                    fill="toself", fillcolor="rgba(139,92,246,.1)",
                    line=dict(color="#8b5cf6",width=2),
                    marker=dict(color="#8b5cf6",size=5)))
                fig_r.update_layout(**_plotly_defaults(),
                    polar=dict(bgcolor="rgba(21,30,45,.8)",
                               radialaxis=dict(visible=True, range=[0,100],
                                              gridcolor="rgba(255,255,255,.05)",
                                              tickfont=dict(color="#4a5a72",size=8)),
                               angularaxis=dict(gridcolor="rgba(255,255,255,.05)",
                                                tickfont=dict(color="#8fa0b8",size=9))),
                    title=dict(text="9-Axis Modality Radar", font=dict(color="#dde5f0",size=12)),
                    height=340)
                st.plotly_chart(fig_r, use_container_width=True)

            with ch2:
                # Stacked bar: per-modality contributions
                modalities_present = []
                mania_vals = []; depr_vals = []; mixed_vals = []

                if result.has_facial:
                    modalities_present.append("Facial")
                    mania_vals.append(result.modality_scores.get("Facial Mania", 0))
                    depr_vals.append(result.modality_scores.get("Facial Depr", 0))
                    mixed_vals.append(result.modality_scores.get("Facial Mixed", 0))
                if result.has_text:
                    modalities_present.append("Text")
                    mania_vals.append(result.modality_scores.get("Text Mania", 0))
                    depr_vals.append(result.modality_scores.get("Text Depr", 0))
                    mixed_vals.append(result.modality_scores.get("Text Mixed", 0))
                if result.has_questionnaire:
                    modalities_present.append("Questionnaire")
                    mania_vals.append(result.modality_scores.get("MDQ (Mania)", 0))
                    depr_vals.append(result.modality_scores.get("PHQ-9 (Depr)", 0))
                    mixed_vals.append(result.modality_scores.get("ALS (Lability)", 0))

                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(name="Mania",      x=modalities_present, y=mania_vals, marker_color="#f87171"))
                fig_c.add_trace(go.Bar(name="Depression", x=modalities_present, y=depr_vals,  marker_color="#60a5fa"))
                fig_c.add_trace(go.Bar(name="Mixed",      x=modalities_present, y=mixed_vals, marker_color="#a78bfa"))
                fig_c.update_layout(**_plotly_defaults(), barmode="group",
                    title=dict(text="Scores by Modality", font=dict(color="#dde5f0",size=12)),
                    yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,.04)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"), height=340)
                st.plotly_chart(fig_c, use_container_width=True)

            # Clinical summary
            import re
            clean_s = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", result.clinical_summary)
            st.markdown(f'<div class="card" style="border-color:rgba(45,212,191,.2)"><h3>🩺 Clinical Summary</h3>'
                        f'<p style="color:#b8c8de;font-size:.92rem;line-height:1.8;margin:0">{clean_s}</p></div>',
                        unsafe_allow_html=True)

            if result.red_flags:
                flags_html = "".join(f'<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04);'
                                     f'font-size:.85rem;color:#fca5a5">⚠ {f}</div>'
                                     for f in result.red_flags)
                st.markdown(f'<div class="card" style="border-color:rgba(244,63,94,.3)">'
                            f'<h3 style="color:#f87171!important">🔴 Critical Flags</h3>{flags_html}</div>',
                            unsafe_allow_html=True)

            recs_html = ""
            for i, rec in enumerate(result.recommendations, 1):
                color = "#f87171" if "URGENT" in rec or "IMMEDIATE" in rec else \
                        "#fbbf24" if "crisis" in rec.lower() else "#b8c8de"
                recs_html += (f'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);'
                              f'font-size:.87rem;color:{color};line-height:1.6"><b>{i}.</b> {rec}</div>')
            st.markdown(f'<div class="card"><h3>✅ Recommendations</h3>{recs_html}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5  ·  REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="card"><h3>📄 Export Screening Report</h3>'
                '<p style="color:#8fa0b8;font-size:.84rem;margin:0">'
                'Generate a clinical-grade PDF report for your mental health provider.</p></div>',
                unsafe_allow_html=True)

    xc1, xc2 = st.columns(2)
    with xc1: rpt_name      = st.text_input("Patient name", value=st.session_state.patient_name or "Anonymous")
    with xc2: rpt_clinician = st.text_input("Clinician",    placeholder="Dr. ...")
    rpt_dob = st.text_input("Date of birth",                placeholder="DD/MM/YYYY")

    if st.button("📥  Generate PDF Report", use_container_width=True):
        if not st.session_state.screening_result:
            st.warning("Compute the composite screening result first (📊 Results tab).")
        else:
            with st.spinner("Generating PDF..."):
                from modules.report_generator import generate_pdf_report
                pdf_bytes = generate_pdf_report(
                    st.session_state.screening_result,
                    patient_name=rpt_name or "Anonymous",
                    dob=rpt_dob or "Not provided",
                    clinician=rpt_clinician or "Not specified",
                )
            if pdf_bytes:
                sid = st.session_state.screening_result.session_id
                st.download_button("⬇️  Download PDF", pdf_bytes,
                                   f"bpdisdet_v2_{sid}.pdf", "application/pdf",
                                   use_container_width=True)
                st.success("✓ PDF ready. Click above to download.")
            else:
                st.error("PDF generation failed. Ensure fpdf2 is installed.")

    st.divider()

    if st.session_state.screening_result:
        result = st.session_state.screening_result
        export = {
            "session_id":  result.session_id,
            "timestamp":   datetime.fromtimestamp(result.timestamp).isoformat(),
            "patient":     {"name": st.session_state.patient_name,
                            "age":  st.session_state.patient_age,
                            "gender": st.session_state.patient_gender},
            "composite_scores": {
                "mania":       round(result.composite_mania, 2),
                "depression":  round(result.composite_depression, 2),
                "mixed":       round(result.composite_mixed, 2),
                "instability": round(result.affective_instability, 2),
            },
            "classification": {
                "dominant_state": result.dominant_state,
                "overall_risk":   result.overall_risk,
                "confidence_pct": round(result.confidence_pct, 1),
            },
            "modalities": {
                "facial":        result.has_facial,
                "text":          result.has_text,
                "questionnaire": result.has_questionnaire,
            },
            "modality_scores":  result.modality_scores,
            "red_flags":        result.red_flags,
            "recommendations":  result.recommendations,
        }
        st.download_button("⬇️  Export JSON", json.dumps(export, indent=2),
                           f"bpdisdet_v2_{result.session_id}.json",
                           "application/json", use_container_width=True)

    st.divider()
    st.markdown("""<div style="font-size:.78rem;color:#4a5a72;text-align:center;line-height:2.2;padding:1rem">
  <b style="color:#8fa0b8">bpdisdet v2</b> · Multimodal Mental Health Screening<br>
  OpenCV · Anthropic Claude · Streamlit · No TensorFlow · Cloud Ready<br>
  Supporting UN SDG-3: Good Health and Well-Being<br>
  <span style="color:#4ade80">Screening tool only · Not a clinical diagnostic instrument</span><br>
  Crisis: <b style="color:#dde5f0">iCall 9152987821</b> · <b style="color:#dde5f0">Vandrevala 1860-2662-345</b>
</div>""", unsafe_allow_html=True)