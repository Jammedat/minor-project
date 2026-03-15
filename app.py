"""
FaceAttend — Biometric Attendance System
Teacher Flow:
  1. Sign Up / Log In
  2. Select Subject
  3. Start Live Attendance (students come one by one)
  4. Stop Attendance
  5. Export Report
  + Student Management (register, enroll, view, edit)
"""

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import cv2
from PIL import Image
import random
import threading

import auth
import face_db
import attendance
import embedder

TODAY_STR = datetime.datetime.now().strftime("%Y-%m-%d")

st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── VARIABLES ── */
:root {
  --blue:        #2563eb;
  --blue-dark:   #1d4ed8;
  --blue-light:  #3b82f6;
  --blue-glow:   rgba(37,99,235,.15);
  --blue-faint:  rgba(37,99,235,.08);
  --green:       #16a34a;
  --green-bg:    rgba(22,163,74,.1);
  --green-bdr:   rgba(22,163,74,.28);
  --red:         #dc2626;
  --red-bg:      rgba(220,38,38,.1);
  --red-bdr:     rgba(220,38,38,.28);
  --amber:       #d97706;
  --amber-bg:    rgba(217,119,6,.1);
  --amber-bdr:   rgba(217,119,6,.28);
  --purple:      #7c3aed;
  --purple-bg:   rgba(124,58,237,.1);
  --purple-bdr:  rgba(124,58,237,.28);
  --font:        'DM Sans', sans-serif;
  --mono:        'DM Mono', monospace;
  --r:           12px;
  --r-lg:        18px;
}

/* ── Light mode (default) ── */
:root {
  --bg:       #f0f4ff;
  --bg2:      #ffffff;
  --bg3:      #e8eef8;
  --bdr:      #dde3f0;
  --bdr2:     #c6cfe4;
  --tx:       #0f172a;
  --tx2:      #1e293b;
  --tx3:      #475569;
  --tx4:      #94a3b8;
  /* Sidebar always dark regardless of page mode */
  --sb-bg:    #1a2540;
  --sb-bg2:   #0f1829;
  --sb-tx:    #e2e8f0;
  --sb-sub:   #64748b;
  --sb-hover: rgba(255,255,255,.06);
}

/* ── Dark mode ── */
@media (prefers-color-scheme: dark) {
  :root {
    --bg:   #0b1120;
    --bg2:  #131e35;
    --bg3:  #0f1829;
    --bdr:  rgba(255,255,255,.07);
    --bdr2: rgba(255,255,255,.13);
    --tx:   #f1f5f9;
    --tx2:  #cbd5e1;
    --tx3:  #64748b;
    --tx4:  #334155;
  }
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: var(--font) !important; }
.stApp { background: var(--bg) !important; font-family: var(--font) !important; }
.stApp * { font-family: var(--font) !important; }
[data-testid="stAppViewContainer"] > section.main { background: var(--bg) !important; }
.block-container { padding: 1.8rem 2.2rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background-color: #1e3a5f !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #ffffff !important; }

/* ── INPUTS ── */
.stTextInput input {
  background: var(--bg2) !important; border: 1.5px solid var(--bdr) !important;
  border-radius: var(--r) !important; color: var(--tx) !important;
  font-size: 14px !important; padding: 10px 14px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px var(--blue-glow) !important; outline: none !important;
}
.stTextInput label, .stSelectbox label, .stSlider label,
.stFileUploader label, .stNumberInput label, .stTextArea label {
  color: var(--tx3) !important; font-size: 11.5px !important;
  font-weight: 600 !important; letter-spacing: .05em !important;
  text-transform: uppercase !important;
}

/* ── SELECT ── */
.stSelectbox > div > div {
  background: var(--bg2) !important; border: 1.5px solid var(--bdr) !important;
  border-radius: var(--r) !important; color: var(--tx) !important;
}
[data-baseweb="popover"] {
  background: var(--bg2) !important; border: 1.5px solid var(--bdr2) !important;
  border-radius: var(--r) !important; box-shadow: 0 8px 30px rgba(0,0,0,.18) !important;
}
[data-baseweb="option"] { background: transparent !important; color: var(--tx) !important; font-size: 13px !important; }
[data-baseweb="option"]:hover { background: var(--blue-faint) !important; }

/* ── BUTTONS ── */
.stButton > button[kind="primary"] {
  background: var(--blue) !important; color: #fff !important;
  border: none !important; border-radius: var(--r) !important;
  font-weight: 700 !important; font-size: 14px !important;
  padding: 10px 24px !important; letter-spacing: -.01em !important;
  box-shadow: 0 4px 14px rgba(37,99,235,.3) !important; transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--blue-dark) !important;
  box-shadow: 0 6px 20px rgba(37,99,235,.45) !important; transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--bg2) !important; color: var(--tx2) !important;
  border: 1.5px solid var(--bdr2) !important; border-radius: var(--r) !important;
  font-weight: 600 !important; font-size: 14px !important; padding: 10px 20px !important;
  transition: all .15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--bg3) !important; border-color: var(--blue) !important; color: var(--tx) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 2px solid var(--bdr) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--tx3) !important;
  border: none !important; border-bottom: 3px solid transparent !important;
  border-radius: 0 !important; padding: 10px 20px 8px !important;
  font-weight: 600 !important; font-size: 13.5px !important;
  transition: all .15s !important; margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] { color: var(--blue) !important; border-bottom-color: var(--blue) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
  background: var(--bg2) !important; border: 1.5px solid var(--bdr) !important;
  border-radius: var(--r-lg) !important; padding: 18px 20px !important;
}
[data-testid="stMetric"] label {
  color: var(--tx3) !important; font-size: 11px !important;
  font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .08em !important;
}
[data-testid="stMetricValue"] {
  color: var(--blue) !important; font-size: 30px !important; font-weight: 800 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border: 1.5px solid var(--bdr) !important; border-radius: var(--r) !important; overflow: hidden !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important; border: 2px dashed var(--bdr2) !important;
  border-radius: var(--r-lg) !important;
}
[data-testid="stFileUploader"] section > button {
  background: var(--blue) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}

/* ── CAMERA ── */
[data-testid="stCameraInput"] video { border-radius: var(--r) !important; }
[data-testid="stCameraInput"] > div > button {
  background: var(--blue) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 700 !important; box-shadow: 0 3px 12px rgba(37,99,235,.35) !important;
}

/* ── PROGRESS ── */
.stProgress > div > div > div { background: var(--blue) !important; border-radius: 4px !important; }

/* ── SCROLLBAR ── */
* { scrollbar-width: thin; scrollbar-color: var(--bdr2) transparent; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: var(--bdr2); border-radius: 4px; }

hr { border: none !important; border-top: 1.5px solid var(--bdr) !important; margin: 1rem 0 !important; }

/* ══ CUSTOM COMPONENTS ══ */

/* Page header */
.fa-ph { display:flex; align-items:center; gap:14px; padding-bottom:1.2rem; border-bottom:2px solid var(--bdr); margin-bottom:1.6rem; }
.fa-ph-icon { width:46px; height:46px; border-radius:13px; background:var(--blue-faint); border:1.5px solid rgba(37,99,235,.25); display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }
.fa-ph-title { font-size:23px; font-weight:800; color:var(--tx); margin:0; letter-spacing:-.03em; }
.fa-ph-sub { font-size:12px; color:var(--tx3); margin-top:3px; }

/* Step badge for teacher flow */
.fa-step { display:inline-flex; align-items:center; gap:8px; background:var(--blue-faint); border:1.5px solid rgba(37,99,235,.2); border-radius:100px; padding:5px 14px 5px 8px; margin-bottom:18px; }
.fa-step-num { width:22px; height:22px; border-radius:50%; background:var(--blue); color:#fff; font-size:11px; font-weight:800; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.fa-step-txt { font-size:12px; font-weight:600; color:var(--blue); letter-spacing:.03em; }

/* Cards */
.fa-card { background:var(--bg2); border:1.5px solid var(--bdr); border-radius:var(--r-lg); padding:22px 24px; margin-bottom:14px; }

/* Banners */
.fa-info    { background:var(--blue-faint); border:1.5px solid rgba(37,99,235,.22); border-radius:var(--r); padding:12px 16px; color:var(--blue-light); font-size:13px; line-height:1.65; margin:8px 0; }
.fa-success { background:var(--green-bg);   border:1.5px solid var(--green-bdr);     border-radius:var(--r); padding:12px 16px; color:#22c55e;            font-size:13px; line-height:1.65; margin:8px 0; }
.fa-warning { background:var(--amber-bg);   border:1.5px solid var(--amber-bdr);     border-radius:var(--r); padding:12px 16px; color:#f59e0b;            font-size:13px; line-height:1.65; margin:8px 0; }
.fa-error   { background:var(--red-bg);     border:1.5px solid var(--red-bdr);       border-radius:var(--r); padding:12px 16px; color:#f87171;            font-size:13px; line-height:1.65; margin:8px 0; }
.fa-purple  { background:var(--purple-bg);  border:1.5px solid var(--purple-bdr);    border-radius:var(--r); padding:12px 16px; color:#a78bfa;            font-size:13px; line-height:1.65; margin:8px 0; }

/* Result cards */
.fa-result       { border-radius:var(--r-lg); padding:28px 20px; text-align:center; margin:6px 0; }
.fa-result-match { background:var(--green-bg); border:2px solid var(--green-bdr); }
.fa-result-fail  { background:var(--red-bg);   border:2px solid var(--red-bdr);   }
.fa-result-idle  { background:var(--bg3);      border:1.5px dashed var(--bdr2);   }
.fa-result-wait  { background:var(--purple-bg);border:2px solid var(--purple-bdr);}
.fa-result-icon  { font-size:38px; margin-bottom:10px; }
.fa-result-name  { font-size:22px; font-weight:800; color:var(--tx); margin:6px 0 4px; letter-spacing:-.02em; }
.fa-result-match .fa-result-name { color:var(--green); }
.fa-result-fail  .fa-result-name { color:var(--red); }
.fa-result-wait  .fa-result-name { color:var(--purple); }
.fa-result-idle  .fa-result-name { color:var(--tx4); font-size:15px; font-weight:500; }
.fa-result-meta  { font-size:13px; color:var(--tx3); line-height:1.9; margin-top:6px; }
.fa-badge        { display:inline-block; border-radius:20px; padding:4px 14px; font-size:11px; font-weight:700; margin-top:12px; letter-spacing:.06em; }
.fa-badge-green  { background:var(--green-bg);  color:var(--green); border:1px solid var(--green-bdr); }
.fa-badge-red    { background:var(--red-bg);    color:var(--red);   border:1px solid var(--red-bdr);   }
.fa-badge-blue   { background:var(--blue-faint);color:var(--blue);  border:1px solid rgba(37,99,235,.25); }
.fa-badge-purple { background:var(--purple-bg); color:var(--purple);border:1px solid var(--purple-bdr); }

/* Confidence bar */
.fa-cbar { margin:10px 0; }
.fa-cbar-lbl { display:flex; justify-content:space-between; font-size:11px; font-weight:600; color:var(--tx3); margin-bottom:5px; letter-spacing:.06em; text-transform:uppercase; }
.fa-cbar-track { height:6px; border-radius:6px; background:var(--bdr); overflow:hidden; }
.fa-cbar-fill  { height:6px; border-radius:6px; transition:width .6s ease; }

/* Attendance list */
.fa-att-list { max-height:320px; overflow-y:auto; }
.fa-att-row  { display:flex; align-items:center; gap:10px; padding:9px 10px; border-radius:8px; border-bottom:1px solid var(--bdr); transition:background .12s; }
.fa-att-row:last-child { border-bottom:none; }
.fa-att-row:hover { background:var(--bg3); }
.fa-att-time { font-size:11px; color:var(--tx4); width:56px; flex-shrink:0; font-family:var(--mono); }
.fa-att-roll { font-size:12px; color:var(--blue); width:48px; flex-shrink:0; font-weight:700; font-family:var(--mono); }
.fa-att-name { font-size:13px; color:var(--tx2); flex:1; font-weight:600; }
.fa-att-tick { color:var(--green); flex-shrink:0; }

/* Live indicator */
.fa-live { display:inline-flex; align-items:center; gap:6px; font-size:11px; color:var(--green); font-weight:700; letter-spacing:.07em; }
.fa-live-dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 1.8s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.5)} }

/* Stat rows */
.fa-stat   { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,.06); font-size:13px; }
.fa-stat:last-child { border-bottom:none; }
.fa-stat-k { color:var(--sb-sub); font-size:12px; }

/* Detail rows */
.fa-drow   { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--bdr); font-size:13px; }
.fa-drow:last-child { border-bottom:none; }
.fa-drow-k { color:var(--tx3); }
.fa-drow-v { color:var(--tx); font-weight:600; }

/* Login card */
.fa-login-card {
  background:var(--bg2); border:1.5px solid var(--bdr2);
  border-radius:var(--r-lg); padding:40px 36px;
  max-width:440px; margin:0 auto;
  box-shadow:0 24px 60px rgba(0,0,0,.12);
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
model_ok, model_err = embedder.model_ready()


def topnav():
    """Render a top navigation bar — fallback when sidebar is not visible."""
    pages = [
        ("attend",   "📸 Attendance"),
        ("students", "👥 Students"),
        ("records",  "📋 Records"),
        ("iris",     "👁️ Iris"),
    ]
    cols = st.columns(len(pages) + 1)
    with cols[0]:
        st.markdown(f"**🎓 {st.session_state.username}**")
    for i, (key, label) in enumerate(pages):
        with cols[i + 1]:
            if st.button(
                label,
                key=f"topnav_{key}",
                use_container_width=True,
                type="primary" if st.session_state.page == key else "secondary",
            ):
                st.session_state.page = key
                st.rerun()
    st.divider()


def ph(icon, title, sub="", step=None):
    topnav()
    step_html = (
        f'<div class="fa-step">'
        f'<div class="fa-step-num">{step}</div>'
        f'<div class="fa-step-txt">STEP {step}</div></div><br>'
        if step else ""
    )
    sub_html = f'<div class="fa-ph-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'{step_html}'
        f'<div class="fa-ph"><div class="fa-ph-icon">{icon}</div>'
        f'<div><div class="fa-ph-title">{title}</div>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )

def banner(msg, kind="info"):
    st.markdown(f'<div class="fa-{kind}">{msg}</div>', unsafe_allow_html=True)

def conf_bar(pct):
    c = "#16a34a" if pct >= 65 else "#d97706" if pct >= 45 else "#dc2626"
    st.markdown(
        f'<div class="fa-cbar">'
        f'<div class="fa-cbar-lbl"><span>CONFIDENCE</span>'
        f'<span style="color:{c};font-weight:700">{pct:.1f}%</span></div>'
        f'<div class="fa-cbar-track">'
        f'<div class="fa-cbar-fill" style="width:{min(pct,100):.0f}%;background:{c}"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

def render_today_list(subject):
    df    = attendance.load_attendance(subject)
    rows  = df[df["Date"] == TODAY_STR] if not df.empty else pd.DataFrame()
    n     = len(rows)
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:8px;margin:16px 0 8px">'
        f'<span style="font-size:14px;font-weight:700;color:var(--tx)">Today\'s Attendance</span>'
        f'<span style="font-size:12px;color:var(--green);font-weight:700">{n} present</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if rows.empty:
        st.markdown(
            '<div style="padding:16px;text-align:center;font-size:13px;color:var(--tx4);'
            'border:1.5px dashed var(--bdr2);border-radius:10px">'
            'No attendance yet today</div>',
            unsafe_allow_html=True,
        )
        return
    html = "".join(
        f'<div class="fa-att-row">'
        f'<span class="fa-att-time">{r.get("Time","—")}</span>'
        f'<span class="fa-att-roll">{r.get("RollNo","—")}</span>'
        f'<span class="fa-att-name">{r.get("Name","—")}</span>'
        f'<span class="fa-att-tick">✓</span></div>'
        for _, r in rows.sort_values("Time", ascending=False).iterrows()
    )
    st.markdown(
        f'<div class="fa-card" style="padding:6px 10px">'
        f'<div class="fa-att-list">{html}</div></div>',
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "logged_in":    False,
    "username":     "",
    "page":         "attend",
    "att_subject":  "",
    "att_state":    None,
    "enroll_imgs":  [],
    "enroll_roll":  "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — always rendered first so it's never blocked by st.stop()
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if st.session_state.logged_in:
        students_all = face_db.get_all_students()
        n_enrolled   = sum(1 for s in students_all if s["n_photos"] >= 3)
        n_total      = len(students_all)

        st.markdown("## 🎓 Face Recognition Attendance")
        st.caption(f"Signed in as **{st.session_state.username}**")
        st.divider()

        st.markdown("**📌 Navigation**")

        if st.button("📸  Take Attendance",
                     use_container_width=True,
                     type="primary" if st.session_state.page == "attend" else "secondary",
                     key="nav_attend"):
            st.session_state.page = "attend"; st.rerun()

        if st.button("👥  Manage Students",
                     use_container_width=True,
                     type="primary" if st.session_state.page == "students" else "secondary",
                     key="nav_students"):
            st.session_state.page = "students"; st.rerun()

        if st.button("📋  Records & Export",
                     use_container_width=True,
                     type="primary" if st.session_state.page == "records" else "secondary",
                     key="nav_records"):
            st.session_state.page = "records"; st.rerun()

        if st.button("👁️  Iris Recognition",
                     use_container_width=True,
                     type="primary" if st.session_state.page == "iris" else "secondary",
                     key="nav_iris"):
            st.session_state.page = "iris"; st.rerun()

        st.divider()
        st.markdown("**⚙️ System**")
        st.markdown(f"Model: {'🟢 Ready' if model_ok else '🔴 Missing'}")
        st.markdown(f"Students: **{n_total}** reg · **{n_enrolled}** enrolled")
        st.caption(datetime.datetime.now().strftime("%d %b %Y"))
        st.divider()

        if st.button("🚪  Sign Out", use_container_width=True, key="nav_signout"):
            st.session_state.logged_in = False
            st.rerun()
    else:
        st.markdown("## 🎓 Face Recognition Attendance")
        st.caption("Please sign in to continue.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOGIN / SIGN UP
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown(
        '<div style="text-align:center;padding:50px 0 0">'
        '<div style="display:inline-flex;align-items:center;gap:9px;'
        'background:var(--blue-faint);border:1.5px solid rgba(37,99,235,.25);'
        'border-radius:100px;padding:8px 22px;margin-bottom:26px">'
        '<span style="font-size:18px">🎓</span>'
        '<span style="font-size:14px;font-weight:800;color:var(--blue);'
        'letter-spacing:-.01em">Face Recognition Attendance</span></div>'
        '<div style="font-size:clamp(26px,4vw,42px);font-weight:800;'
        'color:var(--tx);letter-spacing:-.03em;line-height:1.1;margin-bottom:10px">'
        'Biometric Attendance System</div>'
        '<div style="font-size:13px;color:var(--tx3);margin-bottom:44px;'
        'letter-spacing:.02em">'
        'Face Recognition &nbsp;·&nbsp; Live Liveness Detection &nbsp;·&nbsp; Auto Attendance'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if not model_ok:
        _, cw, _ = st.columns([1,2,1])
        with cw:
            banner("⚠️ <b>Model not loaded.</b> Run the Jupyter notebook and copy model files to <code>data/models/</code>.", "warning")

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown('<div class="fa-login-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;margin-bottom:24px">'
            '<div style="width:52px;height:52px;border-radius:14px;'
            'background:var(--blue-faint);border:1.5px solid rgba(37,99,235,.25);'
            'display:inline-flex;align-items:center;justify-content:center;'
            'font-size:22px;margin-bottom:12px">🎓</div>'
            '<div style="font-size:18px;font-weight:800;color:var(--tx);letter-spacing:-.02em">Teacher Portal</div>'
            '<div style="font-size:12px;color:var(--tx3);margin-top:3px">Sign in to manage attendance</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("form_login"):
                u = st.text_input("Username", placeholder="Enter your username")
                p = st.text_input("Password", type="password", placeholder="Enter your password")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                    if auth.authenticate(u, p):
                        st.session_state.logged_in = True
                        st.session_state.username  = u
                        st.session_state.page      = "attend"
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("form_register"):
                ru  = st.text_input("Username",         placeholder="Choose a username")
                rp  = st.text_input("Password",         type="password", placeholder="Min. 6 characters")
                rp2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Create Account →", use_container_width=True, type="primary"):
                    if rp != rp2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.register_teacher(ru, rp)
                        if ok:
                            st.success(msg + " — please sign in.")
                        else:
                            st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  STEPS 2–5 — TAKE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "attend":

    ph("📸", "Take Attendance", "Live face recognition · Auto liveness · Auto mark")

    if not model_ok:
        banner(f"⚠️ <b>Model not loaded.</b> {model_err}", "warning")
        st.stop()

    n_enrolled = sum(1 for s in face_db.get_all_students() if s["n_photos"] >= 3)
    if n_enrolled == 0:
        banner(
            "⚠️ No students enrolled. Go to <b>👥 Manage Students</b> in the sidebar "
            "to register and enroll students first.",
            "warning",
        )
        st.stop()

    # Check webrtc
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av
        WEBRTC_OK = True
    except ImportError:
        WEBRTC_OK = False

    if not WEBRTC_OK:
        banner(
            "⚠️ <b>Missing package.</b> Run once in terminal then restart:<br>"
            "<code>pip install streamlit-webrtc av</code>",
            "warning",
        )
        st.stop()

    # ── STEP 2: Select subject ────────────────────────────────────────────────
    st.markdown(
        '<div class="fa-step"><div class="fa-step-num">2</div>'
        '<div class="fa-step-txt">SELECT SUBJECT</div></div>',
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3 = st.columns([3, 2, 1])
    with col_s1:
        # Show existing subjects as quick-select + allow typing new one
        existing = attendance.get_all_subjects()
        subj_options = [""] + existing
        subj_sel = st.selectbox(
            "Choose existing subject or type new one below",
            subj_options,
            format_func=lambda x: "— Select subject —" if x == "" else x,
        )
        subj_new = st.text_input("Or type new subject name", placeholder="e.g. Mathematics, Physics...")
        subj = (subj_new.strip() or subj_sel).strip()
        if subj:
            st.session_state.att_subject = subj

    with col_s2:
        match_threshold = st.slider(
            "Match threshold", 0.30, 0.90,
            float(embedder.get_threshold()), 0.05,
            help="Higher = stricter matching",
        )

    with col_s3:
        st.metric("Enrolled", n_enrolled)
        if st.button("➕ Manage Students", use_container_width=True, type="primary"):
            st.session_state.page = "students"
            st.rerun()

    if not subj:
        banner("👆 Select or type a subject name above to begin.", "info")
        st.stop()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── STEPS 3–4: Start / Stop camera ───────────────────────────────────────
    st.markdown(
        '<div class="fa-step"><div class="fa-step-num">3</div>'
        '<div class="fa-step-txt">START LIVE ATTENDANCE · STEP 4: STOP WHEN DONE</div></div>',
        unsafe_allow_html=True,
    )

    # Init shared state
    if st.session_state.att_state is None:
        st.session_state.att_state = {}

    state = st.session_state.att_state
    for k, v in {
        "phase": "challenge", "challenge": "",
        "ch_frames": 0, "blink_open": 0, "blink_close": 0,
        "cx_start": 0.5, "cx_max": 0.5, "cx_min": 0.5,
        "result": None, "result_frames": 0, "last_marked": [],
    }.items():
        if k not in state:
            state[k] = v

    if not state.get("challenge"):
        state["challenge"] = random.choice(["BLINK", "TURN_LEFT", "TURN_RIGHT"])

    CHALLENGE_TEXT  = {"BLINK": "BLINK YOUR EYES", "TURN_LEFT": "TURN HEAD LEFT", "TURN_RIGHT": "TURN HEAD RIGHT"}
    CHALLENGE_EMOJI = {"BLINK": "😉", "TURN_LEFT": "⬅️", "TURN_RIGHT": "➡️"}
    CHALLENGE_HINT  = {
        "BLINK":      "Close both eyes fully then open",
        "TURN_LEFT":  "Turn your head to your left",
        "TURN_RIGHT": "Turn your head to your right",
    }

    _subj_ref   = [subj]
    _thresh_ref = [match_threshold]
    _state_ref  = [state]
    _lock       = threading.Lock()

    def _reset(s):
        s["phase"]       = "challenge"
        s["challenge"]   = random.choice(["BLINK", "TURN_LEFT", "TURN_RIGHT"])
        s["ch_frames"]   = 0
        s["blink_open"]  = 0
        s["blink_close"] = 0
        s["cx_start"]    = 0.5
        s["cx_max"]      = 0.5
        s["cx_min"]      = 0.5
        s["result"]      = None

    class AttendanceProcessor(VideoProcessorBase):
        def __init__(self):
            self._fc = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._ec = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml")

            # Recognition cache: grid_key → (name, roll, conf, matched)
            # Updated by background thread — recv() never waits for it
            self._recog_cache  = {}
            self._recog_lock   = threading.Lock()
            self._recog_busy   = False   # True while bg thread is running
            self._frame_n      = 0

        # ── Background recognition (runs in separate thread) ─────────────────
        def _bg_recognize(self, faces_rgb, thr):
            """Recognize all faces in background. Updates self._recog_cache."""
            new_cache = {}
            for grid_key, face_img in faces_rgb.items():
                emb, _ = embedder.extract_embedding(face_img)
                if emb is not None:
                    res = face_db.find_match(emb, threshold=thr)
                    new_cache[grid_key] = (
                        res["name"], res["roll_no"],
                        res["confidence"], res["matched"])
                else:
                    new_cache[grid_key] = ("Unknown", "", 0.0, False)
            with self._recog_lock:
                self._recog_cache.update(new_cache)
            self._recog_busy = False

        def recv(self, frame_av):
            img  = frame_av.to_ndarray(format="bgr24")
            img  = cv2.flip(img, 1)
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]

            s   = _state_ref[0]
            sj  = _subj_ref[0]
            thr = _thresh_ref[0]
            self._frame_n += 1

            # ── Fast face detection ───────────────────────────────────────────
            # Downsample gray for detection speed
            small   = cv2.resize(gray, (w//2, h//2))
            raw_det = self._fc.detectMultiScale(
                small, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30))
            # Scale coords back up
            if len(raw_det):
                all_faces = [(x*2, y*2, bw*2, bh*2) for (x,y,bw,bh) in raw_det]
            else:
                all_faces = []

            # ── Kick off background recognition every 6 frames ───────────────
            if all_faces and not self._recog_busy and self._frame_n % 6 == 0:
                self._recog_busy = True
                faces_to_recog = {}
                for (fx, fy, fw, fh) in all_faces:
                    gk = (fx//50, fy//50)
                    face_crop = rgb[fy:fy+fh, fx:fx+fw]
                    if face_crop.size > 0:
                        faces_to_recog[gk] = face_crop
                t = threading.Thread(
                    target=self._bg_recognize,
                    args=(faces_to_recog, thr),
                    daemon=True)
                t.start()

            with _lock:
                phase = s["phase"]

                # ── RESULT PHASE ──────────────────────────────────────────────
                if phase == "result":
                    res   = s["result"] or {}
                    ok_r  = res.get("success", False)
                    col   = (22,163,74) if ok_r else (220,38,38)
                    name  = res.get("name","")
                    # Top banner
                    ov = img.copy()
                    cv2.rectangle(ov, (0,0), (w,58), (5,8,20), -1)
                    img = cv2.addWeighted(ov, 0.9, img, 0.1, 0)
                    cv2.putText(img, f"{'PRESENT' if ok_r else 'UNKNOWN'}: {name}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.95, col, 2, cv2.LINE_AA)
                    # Draw boxes
                    for (fx,fy,fw,fh) in all_faces:
                        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh), col, 2)
                    s["result_frames"] -= 1
                    if s["result_frames"] <= 0:
                        _reset(s)

                # ── CHALLENGE PHASE ───────────────────────────────────────────
                else:
                    ch = s["challenge"]

                    if all_faces:
                        primary = max(all_faces, key=lambda f: f[2]*f[3])
                        px, py, pw, ph_ = [int(v) for v in primary]
                        pcx = (px + pw/2) / max(w, 1)

                        # Draw ALL faces with cached names (non-blocking read)
                        with self._recog_lock:
                            cache_snap = dict(self._recog_cache)

                        for (fx,fy,fw,fh) in all_faces:
                            fx,fy,fw,fh = int(fx),int(fy),int(fw),int(fh)
                            is_primary  = (fx==px and fy==py)
                            gk          = (fx//50, fy//50)
                            info        = cache_snap.get(gk)

                            if info:
                                name_r, roll_r, conf_r, matched_r = info
                            else:
                                name_r, matched_r = "...", False

                            # Colors
                            if is_primary:
                                bc = (37,99,235)    # blue = liveness subject
                            elif matched_r:
                                bc = (22,163,74)    # green = recognised
                            else:
                                bc = (220,38,38)    # red = unknown

                            cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh), bc, 2)

                            # Name tag above box
                            tag = name_r if (matched_r or name_r=="...") else "Unknown"
                            (tw,th),_ = cv2.getTextSize(
                                tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                            ty = max(fy-24, 0)
                            cv2.rectangle(img,(fx,ty),(fx+tw+8,ty+th+8), bc,-1)
                            cv2.putText(img, tag,(fx+4,ty+th+3),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.52,
                                        (255,255,255),1,cv2.LINE_AA)

                        # ── Eye detection on primary (fast — small ROI) ───────
                        upper  = gray[py:py+int(ph_*.5), px:px+pw]
                        eyes   = self._ec.detectMultiScale(
                            upper, 1.15, 2, minSize=(8,8))
                        n_eyes = len(eyes)

                        s["ch_frames"] += 1
                        done = False

                        if ch == "BLINK":
                            if n_eyes >= 1: s["blink_open"]  += 1
                            else:           s["blink_close"] += 1
                            if s["blink_close"] >= 1 and s["blink_open"] >= 2:
                                done = True

                        elif ch == "TURN_LEFT":
                            if s["ch_frames"] <= 4:
                                s["cx_start"] = s["cx_max"] = pcx
                            else:
                                if pcx > s["cx_max"]: s["cx_max"] = pcx
                                if s["cx_max"] - s["cx_start"] >= 0.06: done = True

                        elif ch == "TURN_RIGHT":
                            if s["ch_frames"] <= 4:
                                s["cx_start"] = s["cx_min"] = pcx
                            else:
                                if pcx < s["cx_min"]: s["cx_min"] = pcx
                                if s["cx_start"] - s["cx_min"] >= 0.06: done = True

                        if s["ch_frames"] > 180: _reset(s)  # 6s timeout

                        if done:
                            # Use cached recognition if available, else quick run
                            gk   = (px//50, py//50)
                            with self._recog_lock:
                                info = self._recog_cache.get(gk)
                            if info:
                                name_r, roll_r, conf_r, matched_r = info
                            else:
                                face_crop = rgb[py:py+ph_, px:px+pw]
                                emb, _ = embedder.extract_embedding(face_crop)
                                if emb is not None:
                                    res2 = face_db.find_match(emb, threshold=thr)
                                    name_r, roll_r = res2["name"], res2["roll_no"]
                                    conf_r, matched_r = res2["confidence"], res2["matched"]
                                else:
                                    name_r,roll_r,conf_r,matched_r = "Unclear","",0,False

                            if matched_r:
                                marked, _ = attendance.mark_attendance(
                                    sj, roll_r, name_r, method="face")
                                result = {"success":True, "name":name_r,
                                          "roll":roll_r, "conf":conf_r, "marked":marked}
                            else:
                                result = {"success":False, "name":name_r or "Unknown",
                                          "roll":"", "conf":conf_r, "marked":False}

                            s["result"]        = result
                            s["phase"]         = "result"
                            s["result_frames"] = 60   # 2s result display
                            if result.get("success") and result.get("marked"):
                                s["last_marked"].append(
                                    f"{result['name']} ({result['roll']})")
                                s["last_marked"] = s["last_marked"][-15:]

                    # ── Bottom instruction bar ────────────────────────────────
                    ov = img.copy()
                    cv2.rectangle(ov,(0,h-62),(w,h),(5,8,20),-1)
                    img = cv2.addWeighted(ov,0.9,img,0.1,0)
                    cv2.putText(img, CHALLENGE_TEXT.get(ch,""),
                                (10,h-18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.82,(147,130,246),2,cv2.LINE_AA)
                    cv2.putText(img, CHALLENGE_HINT.get(ch,""),
                                (10,h-4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.40,(100,116,139),1,cv2.LINE_AA)
                    # Face count
                    if all_faces:
                        n_f = len(all_faces)
                        cv2.putText(img, f"{n_f} face{'s' if n_f>1 else ''}",
                                    (w-110,26), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.58,(22,163,74),2,cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")

    _subj_ref[0]   = subj
    _thresh_ref[0] = match_threshold

    # Layout
    col_cam, col_right = st.columns([1.25, 1], gap="large")

    with col_cam:
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            '<span style="font-size:15px;font-weight:700;color:var(--tx)">Live Camera</span>'
            '<span class="fa-live"><span class="fa-live-dot"></span>LIVE</span></div>',
            unsafe_allow_html=True,
        )
        banner(
            f"📚 Subject: <b>{subj}</b> &nbsp;·&nbsp; "
            "Camera will instruct each student automatically. "
            "Press <b>STOP</b> on the widget when attendance is complete.",
            "info",
        )
        webrtc_streamer(
            key=f"attend_{subj}",
            video_processor_factory=AttendanceProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_right:
        # Result panel
        res   = state.get("result")
        phase = state.get("phase", "challenge")
        ch    = state.get("challenge", "")

        if phase == "result" and res:
            if res.get("success"):
                badge = "MARKED" if res.get("marked") else "ALREADY MARKED"
                st.markdown(
                    f'<div class="fa-result fa-result-match">'
                    f'<div class="fa-result-icon">✅</div>'
                    f'<div class="fa-result-name">{res["name"]}</div>'
                    f'<div class="fa-result-meta">'
                    f'Roll No. <b>{res["roll"]}</b><br>'
                    f'{"✅ Attendance marked successfully!" if res["marked"] else "⚠️ Already marked today."}'
                    f'<br><br>⏳ <i>Next student please…</i></div>'
                    f'<div class="fa-badge fa-badge-green">{badge}</div></div>',
                    unsafe_allow_html=True,
                )
                conf_bar(res.get("conf", 0))
                # Speak announcement
                speak = (
                    f"{res['name']} is present. Next student please."
                    if res.get("marked") else f"{res['name']} is already marked present."
                )
                st.components.v1.html(
                    f'<script>'
                    f'window.speechSynthesis.cancel();'
                    f'var u=new SpeechSynthesisUtterance("{speak}");'
                    f'u.rate=0.9; u.pitch=1.0; u.volume=1.0;'
                    f'window.speechSynthesis.speak(u);'
                    f'</script>',
                    height=0,
                )
            else:
                st.markdown(
                    f'<div class="fa-result fa-result-fail">'
                    f'<div class="fa-result-icon">❓</div>'
                    f'<div class="fa-result-name">{res.get("name","Unknown")}</div>'
                    f'<div class="fa-result-meta">'
                    f'Face not recognised.<br>Not enrolled or image unclear.<br><br>'
                    f'⏳ <i>Next student please…</i></div>'
                    f'<div class="fa-badge fa-badge-red">NOT RECOGNISED</div></div>',
                    unsafe_allow_html=True,
                )
                conf_bar(res.get("conf", 0))
                st.components.v1.html(
                    '<script>'
                    'window.speechSynthesis.cancel();'
                    'var u=new SpeechSynthesisUtterance("Face not recognised. Please try again.");'
                    'u.rate=0.9; u.volume=1.0;'
                    'window.speechSynthesis.speak(u);'
                    '</script>',
                    height=0,
                )
        elif ch:
            st.markdown(
                f'<div class="fa-result fa-result-wait">'
                f'<div class="fa-result-icon">{CHALLENGE_EMOJI.get(ch,"🎯")}</div>'
                f'<div class="fa-result-name">{CHALLENGE_TEXT.get(ch,"")}</div>'
                f'<div class="fa-result-meta">'
                f'System is watching automatically.<br>'
                f'<b>No button needed</b> — student just performs the action.</div>'
                f'<div class="fa-badge fa-badge-purple">WATCHING</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fa-result fa-result-idle">'
                '<div class="fa-result-icon" style="opacity:.2">📷</div>'
                '<div class="fa-result-name">Click START on the camera</div>'
                '<div class="fa-result-meta" style="color:var(--tx4)">'
                'Allow camera access when prompted.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── STEP 5: Quick export right here ──────────────────────────────
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:var(--tx3);'
            'letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px">'
            '⬇️ Step 5 · Export This Session</div>',
            unsafe_allow_html=True,
        )
        _df = attendance.load_attendance(subj)
        if not _df.empty:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button("CSV", attendance.export_csv(_df),
                    f"attendance_{subj}_{TODAY_STR}.csv", "text/csv",
                    use_container_width=True, type="primary")
            with ec2:
                st.download_button("Excel", attendance.export_excel(_df),
                    f"attendance_{subj}_{TODAY_STR}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True, type="secondary")
        else:
            st.markdown('<div style="font-size:12px;color:var(--tx4)">No records yet.</div>',
                        unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        render_today_list(subj)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — MANAGE STUDENTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "students":
    ph("👥", "Manage Students", "Register · Enroll face photos · View · Edit · Delete", step=7)

    tab_reg, tab_enroll, tab_view, tab_manage = st.tabs([
        "➕  Register New Student",
        "📸  Enroll Face Photos",
        "📋  View All Students",
        "✏️  Edit / Delete",
    ])

    # ── Register ──────────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown("<br>", unsafe_allow_html=True)
        banner("Fill in the student's details below. After registering, go to <b>Enroll Face Photos</b> to add their face.", "info")
        st.markdown("<br>", unsafe_allow_html=True)
        _, cf, _ = st.columns([1, 2, 1])
        with cf:
            st.markdown('<div class="fa-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:16px;font-weight:700;color:var(--tx);'
                'margin-bottom:18px">New Student Registration</div>',
                unsafe_allow_html=True,
            )
            with st.form("form_student"):
                c1, c2 = st.columns(2)
                with c1:
                    roll = st.text_input("Roll Number *", placeholder="e.g. 101")
                with c2:
                    name = st.text_input("Full Name *",   placeholder="e.g. Aarav Sharma")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Register Student →", type="primary", use_container_width=True):
                    if not roll.strip() or not name.strip():
                        st.error("Both Roll Number and Name are required.")
                    else:
                        ok, msg = face_db.register_student(roll.strip(), name.strip())
                        if ok:
                            st.success(f"✅ {msg} Now go to Enroll Face Photos tab.")
                        else:
                            st.error(f"❌ {msg}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Enroll Face ───────────────────────────────────────────────────────────
    with tab_enroll:
        if not model_ok:
            banner("❌ Model not loaded — cannot extract face embeddings.", "error")
            st.stop()

        students = face_db.get_all_students()
        if not students:
            banner("No students registered yet. Go to <b>Register New Student</b> tab first.", "warning")
        else:
            opts = [f"{s['roll_no']}  —  {s['name']}" for s in students]

            # Keep selected student stable across reruns
            if "enroll_selected" not in st.session_state:
                st.session_state.enroll_selected = opts[0]
            # If saved selection is no longer valid reset it
            if st.session_state.enroll_selected not in opts:
                st.session_state.enroll_selected = opts[0]

            sel = st.selectbox(
                "Select student to enroll",
                opts,
                index=opts.index(st.session_state.enroll_selected),
                key="enroll_selectbox",
            )
            st.session_state.enroll_selected = sel

            roll  = sel.split("  —  ")[0].strip()
            sname = sel.split("  —  ")[1].strip()
            s     = face_db.get_student(roll)
            n     = s["n_photos"]

            # Progress bar
            pct = min(n/5, 1) * 100
            bc  = "#16a34a" if n >= 3 else "#d97706"
            st.markdown(
                f'<div style="margin:8px 0 16px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:12px;color:var(--tx3);margin-bottom:5px">'
                f'<span>{sname}</span>'
                f'<span style="color:{bc};font-weight:700">{n} / 5 photos enrolled</span></div>'
                f'<div style="height:6px;background:var(--bdr);border-radius:6px;overflow:hidden">'
                f'<div style="height:6px;width:{pct:.0f}%;background:{bc};border-radius:6px;transition:width .5s"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if n >= 3:
                banner(f"✅ <b>{sname}</b> is enrolled and ready for attendance. You can still add more photos.", "success")

            banner("Take 3–5 clear, well-lit face photos. Each photo adds an embedding — no retraining needed.", "info")
            st.markdown("<br>", unsafe_allow_html=True)

            etab_cam, etab_upload = st.tabs(["📷  Use Webcam", "📁  Upload Photos"])

            with etab_cam:
                s_live = face_db.get_student(roll)
                cam = st.camera_input(
                    f"Take photo of {sname}",
                    key=f"enroll_cam_{roll}_{s_live['n_photos']}",
                    label_visibility="collapsed",
                )
                if cam:
                    img_rgb = np.array(Image.open(cam).convert("RGB"))
                    with st.spinner("Extracting face embedding…"):
                        emb, msg = embedder.extract_embedding(img_rgb)
                    if emb is None:
                        banner(f"❌ {msg}", "error")
                    else:
                        ok, m = face_db.add_embedding(roll, emb)
                        if ok:
                            nc = face_db.get_student(roll)["n_photos"]
                            banner(f"✅ Photo saved! {sname} now has {nc} photos.", "success")
                            st.rerun()
                        else:
                            banner(f"❌ {m}", "error")

            with etab_upload:
                uploaded = st.file_uploader(
                    f"Upload face photos of {sname}",
                    type=["jpg","jpeg","png"],
                    accept_multiple_files=True,
                    key=f"uploader_{roll}",
                )
                if uploaded:
                    prev = st.columns(min(len(uploaded), 5))
                    imgs = []
                    for i, f in enumerate(uploaded):
                        img = np.array(Image.open(f).convert("RGB"))
                        imgs.append(img)
                        with prev[i % 5]:
                            st.image(img, caption=f"#{i+1}", use_column_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"Enroll {len(imgs)} Photo(s)", type="primary", use_container_width=True):
                        saved, errs = 0, []
                        pb = st.progress(0)
                        for i, img_rgb in enumerate(imgs):
                            emb, msg = embedder.extract_embedding(img_rgb)
                            if emb is None:
                                errs.append(f"#{i+1}: {msg}")
                            else:
                                ok, m = face_db.add_embedding(roll, emb)
                                if ok: saved += 1
                                else:  errs.append(m)
                            pb.progress((i+1)/len(imgs))
                        if errs:
                            st.warning(" | ".join(errs))
                        if saved:
                            nc = face_db.get_student(roll)["n_photos"]
                            banner(f"✅ Enrolled {saved} photo(s). {sname} now has {nc} total.", "success")
                            st.rerun()

    # ── View All ──────────────────────────────────────────────────────────────
    with tab_view:
        students = face_db.get_all_students()
        if not students:
            banner("No students registered yet.", "info")
        else:
            n_ready = sum(1 for s in students if s["n_photos"] >= 3)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Registered", len(students))
            m2.metric("Enrolled (≥3 photos)", n_ready)
            m3.metric("Pending Enrollment", len(students) - n_ready)
            st.markdown("<br>", unsafe_allow_html=True)
            df_s = pd.DataFrame([{
                "Roll No":    s["roll_no"],
                "Name":       s["name"],
                "Photos":     s["n_photos"],
                "Status":     "✅ Ready" if s["n_photos"] >= 3 else f"⏳ Need {3-s['n_photos']} more",
                "Registered": s["registered"],
            } for s in students])
            st.dataframe(df_s, use_container_width=True, hide_index=True)

    # ── Edit / Delete ─────────────────────────────────────────────────────────
    with tab_manage:
        students = face_db.get_all_students()
        if not students:
            banner("No students to manage.", "info")
        else:
            opts = [f"{s['roll_no']}  —  {s['name']}" for s in students]
            sel  = st.selectbox("Select student", opts, key="mgmt_sel")
            roll = sel.split("  —  ")[0].strip()
            s    = face_db.get_student(roll)
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1.4, 1], gap="large")

            with c1:
                rc = "#16a34a" if s["n_photos"] >= 3 else "#d97706"
                st.markdown(
                    f'<div class="fa-card">'
                    f'<div style="font-size:12px;font-weight:700;color:var(--tx3);'
                    f'letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px">Student Details</div>'
                    f'<div class="fa-drow"><span class="fa-drow-k">Roll Number</span>'
                    f'<span class="fa-drow-v" style="color:var(--blue);font-family:var(--mono)">'
                    f'{s["roll_no"]}</span></div>'
                    f'<div class="fa-drow"><span class="fa-drow-k">Full Name</span>'
                    f'<span class="fa-drow-v">{s["name"]}</span></div>'
                    f'<div class="fa-drow"><span class="fa-drow-k">Photos Enrolled</span>'
                    f'<span class="fa-drow-v" style="color:{rc}">'
                    f'{{s["n_photos"]}} — {{("Ready" if s["n_photos"]>=3 else "Need "+str(3-s["n_photos"])+" more")}}'
                    f'</span></div>'
                    f'<div class="fa-drow"><span class="fa-drow-k">Registered On</span>'
                    f'<span class="fa-drow-v" style="color:var(--tx3)">{s["registered"]}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown('<div class="fa-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:12px;font-weight:700;color:var(--tx3);'
                    'letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px">Actions</div>',
                    unsafe_allow_html=True,
                )
                new_name = st.text_input("Rename student", value=s["name"], key="rename_input")
                if st.button("Save Name", type="primary", use_container_width=True):
                    ok, msg = face_db.update_student(roll, new_name)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Re-enroll (Clear Photos)", use_container_width=True, type="secondary"):
                    ok, msg = face_db.clear_embeddings(roll)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Delete Student", use_container_width=True, type="secondary"):
                    ok, msg = face_db.delete_student(roll)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — RECORDS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "records":
    ph("📋", "Records & Export", "View attendance data · Download CSV or Excel", step=6)

    subjects = attendance.get_all_subjects()
    if not subjects:
        banner("No attendance records yet. Take attendance first.", "info")
        st.stop()

    banner(
        "📥 <b>How to export:</b> Select subject and date range below, "
        "then click <b>Download CSV</b> or <b>Download Excel</b>.",
        "info",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns([2.5, 1.5, 1.5])
    with cf1:
        subj_f = st.selectbox("Subject", ["All Subjects"] + subjects)
    with cf2:
        date_from = st.date_input("From", value=datetime.date.today() - datetime.timedelta(days=30))
    with cf3:
        date_to   = st.date_input("To",   value=datetime.date.today())

    df = (attendance.load_all_attendance() if subj_f == "All Subjects"
          else attendance.load_attendance(subj_f))

    if df.empty:
        banner("No records found.", "info")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.Timestamp(date_from)) & (df["Date"] <= pd.Timestamp(date_to))]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    if df.empty:
        banner("No records in selected date range.", "info")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",   len(df))
    m2.metric("Unique Students", df["RollNo"].nunique())
    m3.metric("Subjects",        df["Subject"].nunique() if "Subject" in df.columns else 1)
    m4.metric("Days Covered",    df["Date"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    # Big download buttons
    dc1, dc2, dc3 = st.columns([2, 2, 1])
    with dc1:
        st.download_button(
            "⬇️  Download as CSV",
            data=attendance.export_csv(df),
            file_name=f"attendance_{subj_f}_{date_from}_{date_to}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with dc2:
        st.download_button(
            "⬇️  Download as Excel (.xlsx)",
            data=attendance.export_excel(df),
            file_name=f"attendance_{subj_f}_{date_from}_{date_to}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="secondary",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    tab_full, tab_summary = st.tabs(["Full Records", "Per-Student Summary"])

    with tab_full:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_summary:
        summary = (df.groupby(["RollNo","Name"]).size()
                   .reset_index(name="Days Present")
                   .sort_values("Days Present", ascending=False))
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  IRIS RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "iris":
    ph("👁️", "Iris Recognition", "Secondary biometric verification")

    banner(
        "👁️ <b>Iris Recognition:</b> Hold the eye 25–40 cm from camera. "
        "Good lighting required. Remove glasses if possible.",
        "purple",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    subj_i = st.text_input("Subject", placeholder="e.g. Physics")
    st.markdown("<br>", unsafe_allow_html=True)

    if not subj_i.strip():
        banner("Enter a subject name above.", "info")
        st.stop()

    col_cam, col_right = st.columns([1.1, 1], gap="large")
    with col_cam:
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            '<span style="font-size:15px;font-weight:700;color:var(--tx)">Iris Camera</span>'
            '<span class="fa-live"><span class="fa-live-dot" style="background:var(--purple)"></span>'
            '<span style="color:var(--purple)">IRIS SCAN</span></span></div>',
            unsafe_allow_html=True,
        )
        iris_img = st.camera_input("Capture iris", label_visibility="collapsed",
                                   key=f"iris_{subj_i.strip()}")

    with col_right:
        iris_slot = st.empty()
        iris_slot.markdown(
            '<div class="fa-result fa-result-idle">'
            '<div class="fa-result-icon" style="opacity:.2">👁️</div>'
            '<div class="fa-result-name">Waiting for iris scan</div></div>',
            unsafe_allow_html=True,
        )
        if iris_img:
            img_rgb  = np.array(Image.open(iris_img).convert("RGB"))
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (9,9), 2)
            circles  = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 80,
                                         param1=50, param2=30, minRadius=20, maxRadius=90)
            draw = img_rgb.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                cx2, cy2, r2 = circles[0][0]
                cv2.circle(draw, (int(cx2),int(cy2)), int(r2),    (124,58,237), 2)
                cv2.circle(draw, (int(cx2),int(cy2)), int(r2)//3, (124,58,237), 1)
                cv2.circle(draw, (int(cx2),int(cy2)), 3,          (124,58,237), -1)
            with col_cam:
                st.image(draw, use_column_width=True, clamp=True)

            if circles is not None:
                iris_slot.markdown(
                    f'<div class="fa-result fa-result-wait" style="border-color:var(--purple-bdr)">'
                    f'<div class="fa-result-icon">👁️</div>'
                    f'<div class="fa-result-name" style="color:var(--purple)">Iris Detected</div>'
                    f'<div class="fa-result-meta">Radius: {r2}px · Centre: ({cx2}, {cy2})</div>'
                    f'<div class="fa-badge fa-badge-purple">IRIS FOUND</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                iris_slot.markdown(
                    '<div class="fa-result fa-result-fail">'
                    '<div class="fa-result-icon">🚫</div>'
                    '<div class="fa-result-name">No Iris Detected</div>'
                    '<div class="fa-result-meta">Move closer. Ensure good lighting.</div>'
                    '<div class="fa-badge fa-badge-red">NOT DETECTED</div></div>',
                    unsafe_allow_html=True,
                )