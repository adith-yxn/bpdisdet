"""
auth.py  ·  bpdisdet v4
════════════════════════
Professional authentication system:
  • Clean login page — NO credentials shown
  • Full registration form with validation
  • SHA-256 password hashing
  • JSON-file user persistence
  • Brute-force lockout (5 attempts → 60s lock)
  • Password strength meter
  • Role selection (patient / clinician / researcher)
  • Profile: full name, email, org, phone, DOB
"""

import hashlib, json, time, re, os
from datetime import datetime
from pathlib import Path
import streamlit as st

# ── User store path ────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent.parent
_USER_FILE = _HERE / "data" / "users.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    try:
        if _USER_FILE.exists():
            with open(_USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_users(db: dict):
    _USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


ROLE_PERMISSIONS = {
    "admin":      ["text","facial","questionnaire","results","report","admin_panel"],
    "clinician":  ["text","facial","questionnaire","results","report"],
    "patient":    ["text","facial","questionnaire","results","report"],
    "researcher": ["text","facial","questionnaire","results","report"],
}

ROLE_LABELS = {
    "patient":    ("Patient / Self-Screening",  "👤"),
    "clinician":  ("Clinician / Healthcare Pro","🩺"),
    "researcher": ("Researcher / Academic",     "🔬"),
}


# ── Session state init ─────────────────────────────────────────────────────────
def init_auth_state():
    defaults = {
        "authenticated":  False,
        "username":       "",
        "user_role":      "",
        "display_name":   "",
        "avatar_letter":  "U",
        "user_email":     "",
        "user_org":       "",
        "user_theme":     "dark",
        "login_attempts": 0,
        "locked_until":   0,
        "auth_page":      "login",   # "login" | "register"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_locked() -> bool:
    return time.time() < st.session_state.get("locked_until", 0)


# ── Credential verification ────────────────────────────────────────────────────
def verify_login(username: str, password: str) -> tuple[bool, str]:
    if is_locked():
        rem = int(st.session_state.locked_until - time.time())
        return False, f"Account temporarily locked. Try again in {rem}s."

    username = username.strip().lower()
    db = _load_users()
    user = db.get(username)

    if not user or _sha256(password) != user.get("hash",""):
        st.session_state.login_attempts += 1
        if st.session_state.login_attempts >= 5:
            st.session_state.locked_until = time.time() + 60
            return False, "Too many failed attempts. Account locked for 60 seconds."
        left = 5 - st.session_state.login_attempts
        return False, f"Incorrect username or password. ({left} attempt{'s' if left!=1 else ''} remaining)"

    # ── Success ────────────────────────────────────────────────────────
    st.session_state.authenticated  = True
    st.session_state.username        = username
    st.session_state.user_role       = user.get("role","patient")
    st.session_state.display_name    = user.get("display_name", username.title())
    st.session_state.avatar_letter   = user.get("avatar_letter", username[0].upper())
    st.session_state.user_email      = user.get("email","")
    st.session_state.user_org        = user.get("org","")
    st.session_state.user_theme      = user.get("theme","dark")
    st.session_state.login_attempts  = 0
    st.session_state.locked_until    = 0
    return True, "Login successful."


def register_user(form: dict) -> tuple[bool, str]:
    """Validate and create a new user account."""
    username = form["username"].strip().lower()
    password = form["password"]
    confirm  = form["confirm"]
    email    = form["email"].strip().lower()
    name     = form["full_name"].strip()
    role     = form["role"]

    # ── Validation ─────────────────────────────────────────────────────
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-z0-9_\.]+$", username):
        return False, "Username may only contain letters, numbers, underscores, dots."
    if not name or len(name) < 2:
        return False, "Please enter your full name."
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    ok, msg = _check_password_strength(password)
    if not ok:
        return False, msg

    db = _load_users()
    if username in db:
        return False, "Username already taken. Please choose another."
    if any(u.get("email","") == email for u in db.values()):
        return False, "An account with this email already exists."

    # ── Create user ─────────────────────────────────────────────────────
    db[username] = {
        "hash":         _sha256(password),
        "role":         role,
        "display_name": name,
        "email":        email,
        "org":          form.get("org","").strip(),
        "dob":          form.get("dob",""),
        "phone":        form.get("phone","").strip(),
        "created_at":   datetime.now().isoformat(),
        "theme":        "dark",
        "avatar_letter":name[0].upper(),
    }
    _save_users(db)
    return True, "Account created successfully! You can now sign in."


def _check_password_strength(pw: str) -> tuple[bool, str]:
    if not any(c.isupper() for c in pw):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in pw):
        return False, "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in pw):
        return False, "Password must contain at least one number."
    return True, "Strong"


def password_strength_score(pw: str) -> tuple[int, str, str]:
    """Returns (score 0-4, label, color)."""
    score = 0
    if len(pw) >= 8:   score += 1
    if len(pw) >= 12:  score += 1
    if any(c.isupper() for c in pw) and any(c.islower() for c in pw): score += 1
    if any(c.isdigit() for c in pw): score += 1
    if any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in pw): score += 1
    score = min(score, 4)
    labels = ["Very Weak","Weak","Fair","Strong","Very Strong"]
    colors = ["#ef4444","#f97316","#eab308","#22c55e","#06b6d4"]
    return score, labels[score], colors[score]


def logout():
    keys = ["authenticated","username","user_role","display_name","avatar_letter",
            "user_email","user_org","user_theme","login_attempts","locked_until",
            "auth_page","facial_session","text_results","questionnaire_result",
            "screening_result","mdq_answers","phq9_answers","als_answers",
            "demo_text","api_key"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


def save_user_theme(username: str, theme: str):
    db = _load_users()
    if username in db:
        db[username]["theme"] = theme
        _save_users(db)


def has_permission(feature: str) -> bool:
    role = st.session_state.get("user_role","")
    return feature in ROLE_PERMISSIONS.get(role, [])


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_login_page():
    """Professional login page — no credentials exposed."""

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

html, body, [class*="css"], .stApp {
  background: #050a12 !important;
  font-family: 'Inter', sans-serif !important;
  color: #e2e8f0 !important;
}

/* Animated gradient bg */
.stApp {
  background: radial-gradient(ellipse at 20% 50%, rgba(45,212,191,0.04) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 20%, rgba(59,130,246,0.04) 0%, transparent 60%),
              radial-gradient(ellipse at 60% 80%, rgba(139,92,246,0.03) 0%, transparent 60%),
              #050a12 !important;
}
.stApp::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #2dd4bf, #3b82f6, #8b5cf6, #2dd4bf);
  background-size: 200% auto;
  animation: shimmer 4s linear infinite;
  z-index: 9999;
}
@keyframes shimmer { to { background-position: 200% center; } }

[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* Auth card */
.auth-panel {
  background: rgba(10, 18, 32, 0.92);
  border: 1px solid rgba(45,212,191,0.14);
  border-radius: 20px;
  padding: 2.8rem 2.4rem 2.4rem;
  width: 100%;
  backdrop-filter: blur(24px);
  box-shadow: 0 25px 80px rgba(0,0,0,0.6), 0 0 60px rgba(45,212,191,0.04);
}

/* Logo */
.auth-logo {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  color: #2dd4bf;
  text-align: center;
  letter-spacing: -0.04em;
  line-height: 1;
}
.auth-tagline {
  text-align: center;
  color: #475569;
  font-size: 0.78rem;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  margin: 0.3rem 0 0.5rem;
}
.auth-badge {
  text-align: center;
  margin-bottom: 1.8rem;
}
.auth-badge span {
  background: rgba(74,222,128,0.08);
  border: 1px solid rgba(74,222,128,0.2);
  color: #4ade80;
  font-size: 0.64rem;
  padding: 3px 12px;
  border-radius: 20px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.09em;
}

/* Tab switcher */
.auth-tabs {
  display: flex;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 3px;
  margin-bottom: 1.6rem;
  gap: 3px;
}
.auth-tab {
  flex: 1;
  padding: 0.5rem;
  text-align: center;
  border-radius: 7px;
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  color: #64748b;
}
.auth-tab.active {
  background: rgba(45,212,191,0.12);
  color: #2dd4bf;
  border: 1px solid rgba(45,212,191,0.2);
}

/* Inputs */
.stTextInput input, .stTextInput input::placeholder,
.stSelectbox > div > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  color: #e2e8f0 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
}
.stTextInput input:focus {
  border-color: rgba(45,212,191,0.45) !important;
  box-shadow: 0 0 0 3px rgba(45,212,191,0.08) !important;
  background: rgba(45,212,191,0.03) !important;
}
.stTextInput label, .stSelectbox label {
  color: #64748b !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #2dd4bf 0%, #0891b2 100%) !important;
  color: #050a12 !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.96rem !important;
  padding: 0.65rem 2rem !important;
  width: 100% !important;
  transition: all 0.2s !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(45,212,191,0.38) !important;
  filter: brightness(1.06) !important;
}

/* Password strength bar */
.pw-bar-wrap {
  height: 4px;
  border-radius: 2px;
  background: rgba(255,255,255,0.07);
  margin: 4px 0 6px;
  overflow: hidden;
}
.pw-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s, background 0.3s;
}

/* Footer */
.auth-footer {
  text-align: center;
  color: #1e293b;
  font-size: 0.69rem;
  margin-top: 1.6rem;
  line-height: 2;
}
.auth-divider {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 1rem 0;
  color: #1e293b;
  font-size: 0.75rem;
}
.auth-divider::before, .auth-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        # ── Logo & badge ───────────────────────────────────────────────
        st.markdown("""
<div class="auth-panel">
  <div class="auth-logo">bpdisdet</div>
  <div class="auth-tagline">Bipolar Spectrum Disorder Detection System</div>
  <div class="auth-badge"><span>&#10022; UN SDG-3 &middot; Mental Health Access &middot; v4.0</span></div>
</div>
""", unsafe_allow_html=True)

        # ── Page switcher ──────────────────────────────────────────────
        page = st.session_state.get("auth_page", "login")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", use_container_width=True,
                         type="primary" if page == "login" else "secondary"):
                st.session_state.auth_page = "login"
                st.rerun()
        with c2:
            if st.button("Create Account", use_container_width=True,
                         type="primary" if page == "register" else "secondary"):
                st.session_state.auth_page = "register"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.auth_page == "login":
            _render_login_form()
        else:
            _render_register_form()

        st.markdown("""
<div class="auth-footer">
  bpdisdet v4 &nbsp;&middot;&nbsp; For research &amp; screening use only<br>
  Not a clinical diagnostic instrument &nbsp;&middot;&nbsp; All data is private<br>
  Crisis support: iCall 9152987821 &nbsp;&middot;&nbsp; Vandrevala 1860-2662-345
</div>
""", unsafe_allow_html=True)


def _render_login_form():
    username = st.text_input("Username", placeholder="Enter your username", key="li_user")
    password = st.text_input("Password", placeholder="Enter your password",
                              type="password", key="li_pass")

    if is_locked():
        rem = int(st.session_state.locked_until - time.time())
        st.warning(f"Account locked. Try again in {rem}s.")
        return

    if st.button("Sign In  →", key="li_btn"):
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            with st.spinner("Verifying credentials..."):
                time.sleep(0.35)
                ok, msg = verify_login(username, password)
            if ok:
                st.success(f"Welcome back, {st.session_state.display_name}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("""
<div style="text-align:center;margin-top:1rem;color:#334155;font-size:0.8rem">
  Don't have an account? &nbsp;
  <span style="color:#2dd4bf;cursor:pointer" onclick="">Click 'Create Account' above</span>
</div>
""", unsafe_allow_html=True)


def _render_register_form():
    st.markdown('<p style="color:#64748b;font-size:0.82rem;margin:0 0 1rem">Create your free account</p>',
                unsafe_allow_html=True)

    # ── Personal info ──────────────────────────────────────────────────
    full_name = st.text_input("Full Name *", placeholder="Dr. Jane Smith", key="rg_name")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username *", placeholder="janesmith", key="rg_user")
    with col2:
        email = st.text_input("Email Address *", placeholder="jane@hospital.org", key="rg_email")

    # ── Role ───────────────────────────────────────────────────────────
    role_options = {
        "patient":    "Patient / Self-Screening",
        "clinician":  "Clinician / Healthcare Professional",
        "researcher": "Researcher / Academic",
    }
    role_display = st.selectbox("Role *", list(role_options.values()), key="rg_role")
    role_key     = {v: k for k, v in role_options.items()}[role_display]

    # ── Optional fields ────────────────────────────────────────────────
    with st.expander("Additional details (optional)"):
        col3, col4 = st.columns(2)
        with col3:
            org   = st.text_input("Organisation",  placeholder="Hospital / University", key="rg_org")
            phone = st.text_input("Phone Number",  placeholder="+91 98765 43210",       key="rg_phone")
        with col4:
            dob   = st.text_input("Date of Birth", placeholder="DD/MM/YYYY",            key="rg_dob")

    # ── Password ───────────────────────────────────────────────────────
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        password = st.text_input("Password *", placeholder="Min 8 chars",
                                  type="password", key="rg_pass")
    with col6:
        confirm  = st.text_input("Confirm Password *", placeholder="Re-enter password",
                                  type="password", key="rg_conf")

    # Password strength meter
    if password:
        score, label, color = password_strength_score(password)
        pct = int((score / 4) * 100)
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin:2px 0 8px">
  <div class="pw-bar-wrap" style="flex:1">
    <div class="pw-bar-fill" style="width:{pct}%;background:{color}"></div>
  </div>
  <span style="font-size:0.72rem;color:{color};font-family:'JetBrains Mono',monospace;
               white-space:nowrap">{label}</span>
</div>
""", unsafe_allow_html=True)

    # ── Terms ──────────────────────────────────────────────────────────
    agree = st.checkbox(
        "I understand this is a screening tool only and not a clinical diagnostic instrument.",
        key="rg_agree")

    if st.button("Create Account  →", key="rg_btn"):
        if not agree:
            st.warning("Please accept the terms to continue.")
        else:
            form = {
                "username":  username, "password": password, "confirm": confirm,
                "email":     email,    "full_name": full_name, "role": role_key,
                "org":       st.session_state.get("rg_org",""),
                "dob":       st.session_state.get("rg_dob",""),
                "phone":     st.session_state.get("rg_phone",""),
            }
            with st.spinner("Creating your account..."):
                time.sleep(0.4)
                ok, msg = register_user(form)
            if ok:
                st.success(msg)
                st.session_state.auth_page = "login"
                time.sleep(1.2)
                st.rerun()
            else:
                st.error(msg)