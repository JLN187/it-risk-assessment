"""
IT Portfolio Risk Assessment
Prototyp zur Vorhersage von IT-Projektrisikoniveaus mittels ML
Aufbauend auf: Karrenbauer & Breitner (2022)
"""

import streamlit as st
from statistics import mean

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# --------------------------------------------------------------------------------------
# Design-Tokens
# --------------------------------------------------------------------------------------
BG        = "#1b1917"   # warmes Fast-Schwarz
PANEL     = "#231f1d"   # Karten / Panels
PANEL_2   = "#2a2523"   # etwas heller
BORDER    = "#332d29"
TEXT      = "#e6e3e0"
MUTED     = "#8a8178"
ACCENT    = "#cf9b6a"   # warmer Tan/Gold-Akzent (Kategorien, Links, Highlights)
RED       = "#e5484d"   # High-Risk

LEVEL_COLORS = {
    "Low":       "#3ba55d",
    "Medium":    "#d9a441",
    "High":      "#e5484d",
    "Very High": "#c1121f",
    "N/A":       MUTED,
}

# --------------------------------------------------------------------------------------
# Parameter-Schema:  key -> (Label, Optionen, Kurzlabel fuer Radar, Tooltip)
# Reihenfolge innerhalb der Kategorien entspricht dem Design.
# --------------------------------------------------------------------------------------
LEVELS = ["Low", "Medium", "High", "Very High"]

CATEGORIES = {
    "COMPLEXITY": [
        ("team_size", "Team Size", ["1-5", "6-10", "11-20", "20+"], "Team Size",
         "Number of people actively working on the project. 1-5 = small team, 20+ = large team. "
         "Larger teams raise coordination and communication overhead."),
        ("complexity", "Complexity", LEVELS, "Complexity",
         "Overall technical and functional complexity. Low = well-understood, routine work; "
         "Very High = novel, highly intricate requirements."),
        ("integration_complexity", "Integration Complexity", LEVELS, "Integration",
         "Degree to which the project must integrate with existing systems and interfaces. "
         "Low = largely standalone; Very High = many tightly-coupled dependencies."),
    ],
    "EFFICIENCY": [
        ("budget", "Budget", ["$0-50K", "$50K-200K", "$200K-500K", "$500K+"], "Budget",
         "Total budget allocated to the project. Note: in this prototype larger budget tiers are "
         "treated as higher risk (larger scope and financial exposure)."),
        ("budget_utilization", "Budget Utilization", LEVELS, "Budget Util.",
         "How much of the allocated budget is expected to be consumed or is already committed. "
         "High utilization leaves little buffer for change."),
    ],
    "RISK": [
        ("resource_availability", "Resource Availability", LEVELS, "Resources",
         "Availability of required staff, skills and infrastructure. Note: in this prototype a "
         "higher slider position adds to the risk score (see direction note in the docs)."),
        ("technical_debt", "Technical Debt", LEVELS, "Tech Debt",
         "Amount of accumulated shortcuts, legacy code and deferred maintenance the project must "
         "work with. Low = clean codebase; Very High = heavy debt burden."),
        ("team_turnover", "Team Turnover", LEVELS, "Turnover",
         "Expected churn among team members over the project's lifetime. Higher turnover threatens "
         "knowledge continuity."),
    ],
    "STRATEGY": [
        ("past_delivery_success", "Past Delivery Success", LEVELS, "Post Delivery",
         "Track record of comparable past projects delivered by this team/organisation. Note: in "
         "this prototype a higher slider position adds to the risk score (see direction note)."),
    ],
    "URGENCY": [
        ("schedule_pressure", "Schedule Pressure", LEVELS, "Schedule",
         "How tight the deadline is relative to the scope. Low = comfortable schedule; "
         "Very High = aggressive, high-pressure timeline."),
    ],
}

# flache, geordnete Liste aller Parameter (fuer Score + Radar + Ergebnisliste)
FLAT_PARAMS = [(k, lbl, opts, short, tip)
               for group in CATEGORIES.values()
               for (k, lbl, opts, short, tip) in group]
PARAM_LABEL = {k: lbl for (k, lbl, _, _, _) in FLAT_PARAMS}
PARAM_OPTS  = {k: opts for (k, _, opts, _, _) in FLAT_PARAMS}
PARAM_SHORT = {k: short for (k, _, _, short, _) in FLAT_PARAMS}

# --------------------------------------------------------------------------------------
# Score-Logik
# --------------------------------------------------------------------------------------
def value_to_score(key, value):
    """Ordinalwert 1..4 nach Slider-Position, oder None bei N/A."""
    if value is None:
        return None
    return PARAM_OPTS[key].index(value) + 1

def project_score(params):
    """Mittelwert der gesetzten Parameter (N/A ausgeschlossen), oder None."""
    vals = [value_to_score(k, v) for k, v in params.items() if v is not None]
    return round(mean(vals), 2) if vals else None

def score_to_level(score):
    if score is None:
        return "N/A"
    if score < 1.75:
        return "Low"
    if score < 2.5:
        return "Medium"
    if score < 3.25:
        return "High"
    return "Very High"

def portfolio_score(projects):
    scores = [project_score(p["params"]) for p in projects]
    scores = [s for s in scores if s is not None]
    return round(mean(scores), 2) if scores else None

# --------------------------------------------------------------------------------------
# Setup + CSS
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Assessment",
                   layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
    /* ---- Grundflaechen ---- */
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color:{BG}; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
    .block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1500px; }}

    /* ---- Typografie ---- */
    html, body, [class*="css"] {{ color:{TEXT}; }}
    h1 {{ font-weight:600; letter-spacing:-0.01em; }}
    .subtle {{ color:{MUTED}; font-size:0.95rem; }}

    /* ---- Kategorie-Ueberschrift ---- */
    .cat-header {{
        color:{ACCENT}; text-transform:uppercase; letter-spacing:0.14em;
        font-size:0.78rem; font-weight:600; margin:1.7rem 0 0.7rem 0;
    }}
    .param-label {{ font-weight:600; font-size:0.9rem; margin:0.2rem 0 0.15rem 0; }}
    .tick-row {{
        display:flex; justify-content:space-between; color:{ACCENT};
        font-size:0.7rem; opacity:0.75; margin:-0.2rem 0 0.9rem 0;
    }}

    /* ---- Panels ---- */
    .section-card {{
        background:{PANEL}; border:1px solid {BORDER}; border-radius:12px;
        padding:1.4rem 1.6rem; margin-top:0.6rem;
    }}
    .divider {{ height:1px; background:{BORDER}; margin:1.4rem 0; border:none; }}

    /* ---- Buttons ---- */
    .stButton > button {{
        background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
        border-radius:8px; font-weight:500; padding:0.4rem 0.9rem;
    }}
    .stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}

    /* ---- Select-Slider Feinschliff ---- */
    [data-testid="stSelectSlider"] label {{ display:none; }}   /* eigenes Label wird separat gerendert */

    /* ---- Text-Input ---- */
    [data-testid="stTextInput"] input {{
        background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
    }}

    /* ---- Radio (Configure/Results Nav) als Segment-Control ---- */
    div[role="radiogroup"] {{ gap:0.3rem; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Session-State
# --------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("portfolio_created", False)
ss.setdefault("portfolio_name", "New Portfolio")
ss.setdefault("projects", [])        # [{"name": str, "params": {key: value|None}}]
ss.setdefault("view", "Configure")   # "Configure" | "Results"
ss.setdefault("draft_id", 0)         # steigt bei jedem Add Project -> frische Widgets

# --------------------------------------------------------------------------------------
# Sidebar (minimal – der Toggle oben links blendet sie ein/aus)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Portfolios")
    if ss.portfolio_created:
        st.markdown(f"<span class='subtle'>Active:</span> **{ss.portfolio_name}**",
                    unsafe_allow_html=True)
        st.caption(f"{len(ss.projects)} project(s)")
    else:
        st.caption("No portfolio yet.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if st.button("Reset all", use_container_width=True):
        for k in ["portfolio_created", "portfolio_name", "projects", "view", "draft_id"]:
            ss.pop(k, None)
        st.rerun()

# --------------------------------------------------------------------------------------
# Top-Navigation (Configure / Results)
# --------------------------------------------------------------------------------------
nav_col, _ = st.columns([0.35, 0.65])
with nav_col:
    view = st.radio("nav", ["Configure", "Results"],
                    index=0 if ss.view == "Configure" else 1,
                    horizontal=True, label_visibility="collapsed")
    ss.view = view

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ======================================================================================
# EMPTY STATE
# ======================================================================================
def render_empty_state():
    st.markdown(
        f"""
        <div style="text-align:center; padding:7rem 0;">
            <div style="font-size:3.4rem; color:{MUTED}; margin-bottom:0.6rem;">🗎</div>
            <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.3rem;">
                No Portfolio Selected</div>
            <div class="subtle">Create a new portfolio to get started</div>
        </div>
        """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("＋  Create New Portfolio", use_container_width=True):
            ss.portfolio_created = True
            ss.view = "Configure"
            st.rerun()


# ======================================================================================
# CONFIGURE VIEW
# ======================================================================================
def render_slider_row(pid, key, label, opts, tooltip, default_na=True):
    """Ein Parameter: Label + Tooltip, N/A-Checkbox links, Select-Slider rechts, Ticks darunter.
       Gibt den gesetzten Wert (str) oder None (bei N/A) zurueck."""
    st.markdown(f"<div class='param-label'>{label}</div>", unsafe_allow_html=True)
    na_col, sl_col = st.columns([0.07, 0.93])
    with na_col:
        na = st.checkbox("N/A", value=default_na, key=f"na_{pid}_{key}",
                         help="Leave this parameter unset (excluded from the risk score).")
    with sl_col:
        mid = opts[1]  # sinnvoller Default in der Mitte, wenn (noch) N/A
        chosen = st.select_slider(" ", options=opts, value=mid,
                                  key=f"sl_{pid}_{key}", disabled=na,
                                  help=tooltip, label_visibility="collapsed")
    ticks = "".join(f"<span>{o}</span>" for o in opts)
    st.markdown(f"<div class='tick-row'>{ticks}</div>", unsafe_allow_html=True)
    return None if na else chosen


def render_configure():
    st.markdown("## Portfolio Configuration")

    st.markdown("**Portfolio Name**")
    ss.portfolio_name = st.text_input("Portfolio Name", value=ss.portfolio_name,
                                      label_visibility="collapsed")

    # ---- Liste bereits hinzugefuegter Projekte ----
    if ss.projects:
        head_l, head_r = st.columns([0.8, 0.2])
        head_l.markdown("<span class='subtle'>Added Projects</span>", unsafe_allow_html=True)
        head_r.markdown(f"<div style='text-align:right' class='subtle'>{len(ss.projects)} "
                        f"project(s)</div>", unsafe_allow_html=True)
        for i, proj in enumerate(ss.projects):
            row_l, row_r = st.columns([0.95, 0.05])
            row_l.markdown(f"<div style='color:{ACCENT}; padding:0.35rem 0;'>{proj['name']}</div>",
                           unsafe_allow_html=True)
            if row_r.button("🗑", key=f"del_{i}", help="Remove project"):
                ss.projects.pop(i)
                st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ---- aktueller Projekt-Entwurf ----
    pid = ss.draft_id
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        st.markdown("**Project Name**")
        default_name = f"Project {len(ss.projects) + 1}"
        proj_name = st.text_input("Project Name", value=default_name,
                                  key=f"pname_{pid}", label_visibility="collapsed")

        draft = {}
        for cat, params in CATEGORIES.items():
            st.markdown(f"<div class='cat-header'>{cat}</div>", unsafe_allow_html=True)
            for key, label, opts, _short, tip in params:
                draft[key] = render_slider_row(pid, key, label, opts, tip)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Aktionen ----
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    a_col, b_col, _ = st.columns([0.22, 0.22, 0.56])
    with a_col:
        if st.button("＋  Add Project", use_container_width=True):
            ss.projects.append({"name": proj_name, "params": draft})
            ss.draft_id += 1          # frische Widgets fuer das naechste Projekt
            st.rerun()
    with b_col:
        if st.button("🧮  Calculate Results", use_container_width=True):
            if not ss.projects:
                st.warning("Add at least one project before calculating results.")
            else:
                ss.view = "Results"
                st.rerun()


# ======================================================================================
# RESULTS VIEW
# ======================================================================================
def radar(params):
    if not HAS_PLOTLY:
        st.info("Install plotly to see the parameter-profile radar (pip install plotly).")
        return
    theta = [PARAM_SHORT[k] for (k, *_ ) in FLAT_PARAMS]
    r = [value_to_score(k, params.get(k)) or 0 for (k, *_ ) in FLAT_PARAMS]
    theta_closed = theta + [theta[0]]
    r_closed = r + [r[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_closed, theta=theta_closed, fill="toself",
        line=dict(color=RED, width=2), fillcolor="rgba(229,72,77,0.25)"))
    fig.update_layout(
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(range=[0, 4], showticklabels=False, gridcolor=BORDER,
                            linecolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(color=MUTED, size=10))),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30), height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_project_card(proj):
    params = proj["params"]
    score = project_score(params)
    level = score_to_level(score)
    color = LEVEL_COLORS[level]

    # ---- immer sichtbar: Header + kompaktes Raster ----
    def cell(key):
        v = params.get(key)
        v = v if v is not None else "N/A"
        vc = color if v != "N/A" else MUTED
        return (f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;'>"
                f"<span style='color:{MUTED};'>{PARAM_LABEL[key]}</span>"
                f"<span style='color:{vc}; font-weight:600;'>{v}</span></div>")

    compact_keys = ["team_size", "complexity", "integration_complexity",
                    "technical_debt", "team_turnover", "schedule_pressure"]
    grid = ""
    for row_start in range(0, len(compact_keys), 3):
        cols = compact_keys[row_start:row_start + 3]
        grid += "<div style='display:flex; gap:2.5rem;'>"
        grid += "".join(f"<div style='flex:1;'>{cell(k)}</div>" for k in cols)
        grid += "</div>"

    st.markdown(
        f"""
        <div style="background:rgba(229,72,77,0.06); border:1px solid {color}55;
                    border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:1.15rem; font-weight:600;">{proj['name']}</div>
                <div style="color:{MUTED}; font-size:0.85rem;">Risk Score {score if score
                    is not None else '–'}</div>
            </div>
            <div style="color:{MUTED}; font-size:0.85rem; margin:0.1rem 0 0.9rem 0;">
                Risk Level: <span style="color:{color}; font-weight:600;">{level}</span></div>
            {grid}
        </div>
        """, unsafe_allow_html=True)

    # ---- ausklappbar: Radar + vollstaendige Parameterliste ----
    with st.expander("Parameter profile & all parameters"):
        left, right = st.columns([0.5, 0.5])
        with left:
            st.markdown("<div class='cat-header'>PARAMETER PROFILE</div>",
                        unsafe_allow_html=True)
            radar(params)
        with right:
            st.markdown("<div class='cat-header'>ALL PARAMETERS</div>", unsafe_allow_html=True)
            rows = ""
            for key, label, *_ in FLAT_PARAMS:
                v = params.get(key)
                v = v if v is not None else "N/A"
                vc = TEXT if v != "N/A" else MUTED
                rows += (f"<div style='display:flex; justify-content:space-between;"
                         f" padding:0.3rem 0; border-bottom:1px solid {BORDER};'>"
                         f"<span style='color:{MUTED};'>{label}</span>"
                         f"<span style='color:{vc}; font-weight:600;'>{v}</span></div>")
            rows += (f"<div style='display:flex; justify-content:space-between; padding:0.6rem 0"
                     f" 0.1rem 0;'><span style='color:{MUTED};'>Risk Score</span>"
                     f"<span style='color:{color}; font-weight:700;'>{level} · "
                     f"{score if score is not None else '–'}</span></div>")
            st.markdown(rows, unsafe_allow_html=True)


def render_results():
    st.markdown("## Risk Assessment Results")
    st.markdown(f"<div class='subtle'>{ss.portfolio_name}</div>", unsafe_allow_html=True)

    if not ss.projects:
        st.info("No projects to assess yet. Go to **Configure**, add a project, and calculate.")
        return

    p_score = portfolio_score(ss.projects)
    p_level = score_to_level(p_score)
    p_color = LEVEL_COLORS[p_level]

    # ---- Portfolio Risk Summary ----
    st.markdown(
        f"""
        <div class="section-card" style="margin-bottom:1.4rem;">
            <div style="font-size:1.2rem; font-weight:600; margin-bottom:1.1rem;">
                Portfolio Risk Summary</div>
            <div style="display:flex; gap:5rem;">
                <div>
                    <div class="subtle">Total Portfolio Risk</div>
                    <div style="font-size:2rem; font-weight:700; color:{p_color};">
                        {p_level}</div>
                </div>
                <div style="margin-left:auto; text-align:right;">
                    <div class="subtle">Risk Score</div>
                    <div style="font-size:1.6rem; font-weight:600;">
                        {p_score if p_score is not None else '–'}</div>
                </div>
                <div style="text-align:right;">
                    <div class="subtle">Total Projects</div>
                    <div style="font-size:1.6rem; font-weight:600;">{len(ss.projects)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Project Risk Breakdown")
    for proj in ss.projects:
        render_project_card(proj)


# ======================================================================================
# ROUTER
# ======================================================================================
if not ss.portfolio_created:
    render_empty_state()
elif ss.view == "Configure":
    render_configure()
else:
    render_results()
