"""themes.py · bpdisdet v5 — Four professional UI themes."""

THEMES = {
    "dark": {
        "label": "Dark", "icon": "🌙",
        "--bg0": "#070d14", "--bg1": "#0e1828", "--bg2": "#141f30", "--bg3": "#1a2840",
        "--accent": "#2dd4bf", "--accent2": "#3b82f6", "--accent3": "#8b5cf6",
        "--fg": "#e2e8f0", "--fg2": "#8fa0b8", "--fg3": "#3d5070",
        "--border": "rgba(255,255,255,0.06)", "--card-bg": "#141f30",
        "--sidebar-bg": "#0e1828", "--btn-txt": "#070d14",
        "--input-bg": "rgba(26,40,64,0.9)",
        "--plot-bg": "rgba(20,31,48,0.85)", "--grid-c": "rgba(255,255,255,0.07)",
        "--title-c": "#e2e8f0", "--font-c": "#8fa0b8",
    },
    "light": {
        "label": "Light", "icon": "☀️",
        "--bg0": "#f0f4f8", "--bg1": "#e2e8f0", "--bg2": "#ffffff", "--bg3": "#f8fafc",
        "--accent": "#0891b2", "--accent2": "#2563eb", "--accent3": "#7c3aed",
        "--fg": "#0f172a", "--fg2": "#475569", "--fg3": "#94a3b8",
        "--border": "rgba(0,0,0,0.10)", "--card-bg": "#ffffff",
        "--sidebar-bg": "#e2e8f0", "--btn-txt": "#ffffff",
        "--input-bg": "#ffffff",
        "--plot-bg": "rgba(248,250,252,0.95)", "--grid-c": "rgba(0,0,0,0.08)",
        "--title-c": "#0f172a", "--font-c": "#475569",
    },
    "clinical": {
        "label": "Clinical", "icon": "🏥",
        "--bg0": "#f4f7fb", "--bg1": "#e8eef6", "--bg2": "#ffffff", "--bg3": "#eef2f8",
        "--accent": "#1d4ed8", "--accent2": "#0369a1", "--accent3": "#6d28d9",
        "--fg": "#0c1e3c", "--fg2": "#3b5280", "--fg3": "#8098c0",
        "--border": "rgba(29,78,216,0.12)", "--card-bg": "#ffffff",
        "--sidebar-bg": "#dce7f5", "--btn-txt": "#ffffff",
        "--input-bg": "#ffffff",
        "--plot-bg": "rgba(255,255,255,0.95)", "--grid-c": "rgba(29,78,216,0.08)",
        "--title-c": "#0c1e3c", "--font-c": "#3b5280",
    },
    "ocean": {
        "label": "Ocean", "icon": "🌊",
        "--bg0": "#020f1a", "--bg1": "#051624", "--bg2": "#071e30", "--bg3": "#0a2540",
        "--accent": "#06b6d4", "--accent2": "#0284c7", "--accent3": "#a855f7",
        "--fg": "#cce7f5", "--fg2": "#6b9bb8", "--fg3": "#2a4a60",
        "--border": "rgba(6,182,212,0.12)", "--card-bg": "#071e30",
        "--sidebar-bg": "#051624", "--btn-txt": "#020f1a",
        "--input-bg": "rgba(10,37,64,0.9)",
        "--plot-bg": "rgba(7,30,48,0.85)", "--grid-c": "rgba(255,255,255,0.05)",
        "--title-c": "#cce7f5", "--font-c": "#6b9bb8",
    },
}
DEFAULT_THEME = "dark"


def get_theme_vars(key: str) -> dict:
    return THEMES.get(key, THEMES[DEFAULT_THEME])


def get_theme_css(theme_key: str) -> str:
    t = THEMES.get(theme_key, THEMES[DEFAULT_THEME])
    is_light = theme_key in ("light", "clinical")
    sb_thumb = "rgba(0,0,0,0.18)" if is_light else "rgba(255,255,255,0.10)"
    v = "\n".join(f"  {k}: {v};" for k, v in t.items() if k.startswith("--"))

    return f"""
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{ {v} }}

html,body,[class*="css"],.stApp {{
  background: var(--bg0) !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
}}
.stApp {{ background: var(--bg0) !important; }}
.stApp::before {{
  content: '';
  position: fixed; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  z-index: 9999;
}}

[data-testid="stSidebar"] {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}}

.card {{
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.8rem;
  transition: border-color .2s;
}}
.card:hover {{ border-color: var(--accent); opacity: 0.8; }}
.card h3 {{
  font-family: 'Syne', sans-serif !important;
  font-size: 1rem !important;
  color: var(--accent) !important;
  margin: 0 0 .45rem !important;
}}

.tile {{
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 0.6rem;
  text-align: center;
  transition: transform .15s;
}}
.tile:hover {{ transform: translateY(-2px); }}
.tile .val {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.85rem;
  font-weight: 500;
  line-height: 1.1;
}}
.tile .lbl {{
  font-size: 0.67rem;
  color: var(--fg2);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: .07em;
}}

.chip {{ display:inline-block; padding:3px 12px; border-radius:20px;
  font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.05em; }}
.chip-high     {{ background:rgba(244,63,94,.17); color:#f87171; border:1px solid rgba(244,63,94,.35); }}
.chip-moderate {{ background:rgba(245,158,11,.17); color:#f59e0b; border:1px solid rgba(245,158,11,.35); }}
.chip-low      {{ background:rgba(45,212,191,.14); color:var(--accent); border:1px solid var(--accent); opacity:.7; }}
.chip-minimal  {{ background:rgba(59,130,246,.13); color:var(--accent2); border:1px solid var(--accent2); opacity:.7; }}

.stTextInput input, .stTextArea textarea {{
  background: var(--input-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  transition: border-color .2s !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(45,212,191,.1) !important;
}}
.stTextInput label, .stTextArea label, .stSelectbox label {{
  color: var(--fg2) !important;
  font-size: .8rem !important;
  font-weight: 500 !important;
}}

.stButton > button {{
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: var(--btn-txt) !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .87rem !important;
  transition: all .2s !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(45,212,191,.3) !important;
  filter: brightness(1.08) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg1) !important;
  border-radius: 10px;
  padding: 4px;
  border: 1px solid var(--border);
  gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--fg2) !important;
  border-radius: 7px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .84rem !important;
  border: none !important;
  transition: all .15s !important;
}}
.stTabs [aria-selected="true"] {{
  background: var(--card-bg) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: .9rem !important; }}

.stSelectbox > div > div {{
  background: var(--input-bg) !important;
  border-color: var(--border) !important;
  color: var(--fg) !important;
  border-radius: 8px !important;
}}
.stProgress > div > div {{
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 4px !important;
}}

.hero {{ text-align:center; padding:1.6rem 1rem 1rem; }}
.hero h1 {{
  font-family:'Syne',sans-serif !important;
  font-size:clamp(1.8rem,4vw,2.8rem) !important;
  font-weight:800 !important;
  color:var(--accent) !important;
  letter-spacing:-.03em;
  margin:0 !important;
}}
.hero .sub {{ color:var(--fg2); font-size:.87rem; margin-top:.3rem; letter-spacing:.06em; text-transform:uppercase; }}
.badge {{
  display:inline-block;
  background:rgba(74,222,128,.1); border:1px solid rgba(74,222,128,.22);
  color:#4ade80; font-size:.67rem; padding:2px 11px; border-radius:20px;
  font-family:'JetBrains Mono',monospace; letter-spacing:.09em; margin-top:.5rem;
}}
.disclaimer {{
  background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.22);
  border-radius:10px; padding:.7rem 1rem; font-size:.79rem;
  color:#f59e0b; line-height:1.6; margin-bottom:1rem;
}}
.acc-badge {{ display:inline-block; padding:2px 10px; border-radius:20px;
  font-family:'JetBrains Mono',monospace; font-size:.71rem; font-weight:500; }}
.acc-high  {{ background:rgba(34,197,94,.14); color:#4ade80; border:1px solid rgba(34,197,94,.28); }}
.acc-med   {{ background:rgba(245,158,11,.14); color:#fbbf24; border:1px solid rgba(245,158,11,.28); }}
.acc-low   {{ background:rgba(244,63,94,.14);  color:#f87171; border:1px solid rgba(244,63,94,.28); }}
.q-item {{ background:var(--bg3); border:1px solid var(--border);
  border-radius:8px; padding:.8rem 1rem; margin-bottom:.45rem; }}
.q-item .q-text {{ font-size:.88rem; color:var(--fg); line-height:1.5; margin-bottom:.4rem; }}
.q-item .q-cat {{ font-size:.67rem; color:var(--fg3); font-family:'JetBrains Mono',monospace; text-transform:uppercase; }}
.section-divider {{ height:1px; background:var(--border); margin:.9rem 0; }}
.sidebar-brand {{ font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800; color:var(--accent); margin:0; }}
.sidebar-version {{ color:var(--fg3); font-size:.73rem; margin:0 0 8px; }}
.user-profile-bar {{ background:var(--bg3); border:1px solid var(--border); border-radius:10px;
  padding:.7rem .9rem; display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem; }}
.user-avatar-circle {{ width:36px; height:36px; border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:1rem; color:var(--btn-txt); flex-shrink:0; }}
.user-info {{ flex:1; min-width:0; }}
.user-name {{ font-weight:600; font-size:.87rem; color:var(--fg); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.user-role-badge {{ font-size:.67rem; color:var(--fg3); font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:.05em; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:var(--bg0); }}
::-webkit-scrollbar-thumb {{ background:{sb_thumb}; border-radius:3px; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.stDeployButton {{ display:none; }}
"""