"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import streamlit as st
import model_bridge as mb

# --------------------------------------------------------------------------------------
# Design-Tokens
# --------------------------------------------------------------------------------------
BG      = "#1b1917"
PANEL   = "#231f1d"
PANEL_2 = "#2a2523"
BORDER  = "#332d29"
TEXT    = "#e6e3e0"
MUTED   = "#8a8178"
ACCENT  = "#cf9b6a"

LEVEL_COLORS = {
    "Low":      "#3ba55d",
    "Medium":   "#d9a441",
    "High":     "#e5844d",
    "Critical": "#e5484d",
    "N/A":      MUTED,
}

# --------------------------------------------------------------------------------------
# Artefakte laden (einmalig)
# --------------------------------------------------------------------------------------
@st.cache_resource
def _load():
    return mb.load_all()

try:
    MODEL, META, SPEC = _load()
    LOAD_ERROR = None
except Exception as e:                                   # Artefakte fehlen o. Versionskonflikt
    MODEL = META = SPEC = None
    LOAD_ERROR = str(e)

# --------------------------------------------------------------------------------------
# Setup + CSS
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Assessment",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color:{BG}; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
    .block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1500px; }}
    html, body, [class*="css"] {{ color:{TEXT}; }}
    h1 {{ font-weight:600; letter-spacing:-0.01em; }}
    .subtle {{ color:{MUTED}; font-size:0.95rem; }}
    .cat-header {{
        color:{ACCENT}; text-transform:uppercase; letter-spacing:0.14em;
        font-size:0.78rem; font-weight:600; margin:1.7rem 0 0.7rem 0;
    }}
    .param-label {{ font-weight:600; font-size:0.9rem; margin:0.2rem 0 0.15rem 0; }}
    .section-card {{
        background:{PANEL}; border:1px solid {BORDER}; border-radius:12px;
        padding:1.4rem 1.6rem; margin-top:0.6rem;
    }}
    .divider {{ height:1px; background:{BORDER}; margin:1.4rem 0; border:none; }}
    .stButton > button {{
        background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
        border-radius:8px; font-weight:500; padding:0.4rem 0.9rem;
    }}
    .stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
    [data-testid="stTextInput"] input {{ background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER}; }}
    div[role="radiogroup"] {{ gap:0.3rem; }}
</style>
""", unsafe_allow_html=True)

if LOAD_ERROR:
    st.error("Modell-Artefakte konnten nicht geladen werden. Bitte zuerst `masterskript_final.py` "
             "lokal ausfuehren, damit `model_pipeline.joblib` und `feature_defaults.joblib` "
             f"entstehen.\n\nDetails: {LOAD_ERROR}")
    st.stop()

# --------------------------------------------------------------------------------------
# Session-State
# --------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("portfolio_created", False)
ss.setdefault("portfolio_name", "New Portfolio")
ss.setdefault("projects", [])        # [{"name": str, "params": {feature: value|None}}]
ss.setdefault("view", "Configure")
ss.setdefault("draft_id", 0)

MIN_FEATURES = 5

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Portfolios")
    if ss.portfolio_created:
        st.markdown(f"<span class='subtle'>Active:</span> **{ss.portfolio_name}**", unsafe_allow_html=True)
        st.caption(f"{len(ss.projects)} project(s)")
    else:
        st.caption("No portfolio yet.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if st.button("Reset all", use_container_width=True):
        for k in ["portfolio_created", "portfolio_name", "projects", "view", "draft_id"]:
            ss.pop(k, None)
        st.rerun()

# --------------------------------------------------------------------------------------
# Top-Navigation
# --------------------------------------------------------------------------------------
nav_col, _ = st.columns([0.35, 0.65])
with nav_col:
    ss.view = st.radio("nav", ["Configure", "Results"],
                       index=0 if ss.view == "Configure" else 1,
                       horizontal=True, label_visibility="collapsed")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ======================================================================================
# EMPTY STATE
# ======================================================================================
def render_empty_state():
    st.markdown(
        f"""<div style="text-align:center; padding:7rem 0;">
            <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.3rem;">No Portfolio Selected</div>
            <div class="subtle">Create a new portfolio to get started</div></div>""",
        unsafe_allow_html=True)
    _, c2, _ = st.columns([1, 1, 1])
    with c2:
        if st.button("\uFF0B  Create New Portfolio", use_container_width=True):
            ss.portfolio_created = True
            ss.view = "Configure"
            st.rerun()


# ======================================================================================
# CONFIGURE VIEW
# ======================================================================================
def render_feature(pid, feat):
    """Ein Feature: Label + N/A-Checkbox + Control. Gibt Modellwert oder None zurueck."""
    spec = SPEC[feat]
    label = feat.replace("_", " ")
    st.markdown(f"<div class='param-label'>{label}</div>", unsafe_allow_html=True)
    na_col, ctl_col = st.columns([0.08, 0.92])
    with na_col:
        na = st.checkbox("N/A", value=True, key=f"na_{pid}_{feat}",
                         help="Nicht angeben (wird durch einen Standardwert ersetzt).")
    with ctl_col:
        opts = spec["options"]
        if spec["type"] == "nominal":
            choice = st.selectbox(" ", opts, key=f"in_{pid}_{feat}",
                                  disabled=na, label_visibility="collapsed")
        else:
            default = opts[len(opts) // 2]
            choice = st.select_slider(" ", options=opts, value=default,
                                      key=f"in_{pid}_{feat}", disabled=na,
                                      label_visibility="collapsed")
    return None if na else spec["value_map"][choice]


def render_configure():
    st.markdown("## Portfolio Configuration")
    st.markdown("**Portfolio Name**")
    ss.portfolio_name = st.text_input("Portfolio Name", value=ss.portfolio_name,
                                      label_visibility="collapsed")

    if ss.projects:
        st.markdown("<span class='subtle'>Added Projects</span>", unsafe_allow_html=True)
        for i, proj in enumerate(ss.projects):
            row_l, row_r = st.columns([0.95, 0.05])
            row_l.markdown(f"<div style='color:{ACCENT}; padding:0.35rem 0;'>{proj['name']}</div>",
                           unsafe_allow_html=True)
            if row_r.button("\U0001F5D1", key=f"del_{i}", help="Remove project"):
                ss.projects.pop(i)
                st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pid = ss.draft_id
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**Project Name**")
    proj_name = st.text_input("Project Name", value=f"Project {len(ss.projects) + 1}",
                              key=f"pname_{pid}", label_visibility="collapsed")
    st.caption(f"Mindestens {MIN_FEATURES} Merkmale angeben. Je mehr Merkmale gesetzt sind, "
               f"desto zuverlaessiger die Vorhersage.")

    draft = {}
    for crit, feats in mb.KUB_GROUPS.items():
        st.markdown(f"<div class='cat-header'>{crit}</div>", unsafe_allow_html=True)
        for feat in feats:
            draft[feat] = render_feature(pid, feat)
    st.markdown("</div>", unsafe_allow_html=True)

    n_set = sum(v is not None for v in draft.values())
    a_col, b_col, _ = st.columns([0.24, 0.24, 0.52])
    with a_col:
        if st.button(f"\uFF0B  Add Project ({n_set} set)", use_container_width=True):
            if n_set < MIN_FEATURES:
                st.warning(f"Bitte mindestens {MIN_FEATURES} Merkmale setzen (aktuell {n_set}).")
            else:
                ss.projects.append({"name": proj_name, "params": draft})
                ss.draft_id += 1
                st.rerun()
    with b_col:
        if st.button("\U0001F9EE  Calculate Results", use_container_width=True):
            if not ss.projects:
                st.warning("Bitte zuerst mindestens ein Projekt hinzufuegen.")
            else:
                ss.view = "Results"
                st.rerun()


# ======================================================================================
# RESULTS VIEW
# ======================================================================================
def prob_bars(order, proba, pred):
    rows = ""
    for cls, p in zip(order, proba):
        c = LEVEL_COLORS[cls]
        strong = "font-weight:700;" if cls == pred else "opacity:0.85;"
        rows += (
            f"<div style='display:flex; align-items:center; gap:0.8rem; margin:0.25rem 0;'>"
            f"<div style='width:70px; color:{MUTED}; font-size:0.85rem;'>{cls}</div>"
            f"<div style='flex:1; background:{PANEL_2}; border-radius:6px; height:14px;'>"
            f"<div style='width:{p*100:.1f}%; background:{c}; height:14px; border-radius:6px;'></div></div>"
            f"<div style='width:52px; text-align:right; color:{c}; {strong} font-size:0.85rem;'>{p:.1%}</div>"
            f"</div>")
    return rows


def render_project_card(proj):
    order, proba = mb.predict(MODEL, META, proj["params"])
    pred = order[list(proba).index(max(proba))]
    color = LEVEL_COLORS[pred]
    score = mb.expected_score(order, proba)
    n_set = sum(v is not None for v in proj["params"].values())

    st.markdown(
        f"""<div style="background:{PANEL}; border:1px solid {color}66; border-radius:12px;
                 padding:1.1rem 1.3rem; margin-bottom:0.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:1.15rem; font-weight:600;">{proj['name']}</div>
                <div style="color:{color}; font-weight:600;">{pred}</div></div>
            <div style="color:{MUTED}; font-size:0.85rem; margin:0.1rem 0 0.9rem 0;">
                Expected score: {score:.2f} · {n_set} feature(s) set</div>
            {prob_bars(order, proba, pred)}
        </div>""", unsafe_allow_html=True)

    with st.expander("Set parameters"):
        rows = ""
        for feat, v in proj["params"].items():
            if v is not None:
                rows += (f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;"
                         f" border-bottom:1px solid {BORDER};'>"
                         f"<span style='color:{MUTED};'>{feat.replace('_',' ')}</span>"
                         f"<span style='color:{TEXT}; font-weight:600;'>{v}</span></div>")
        st.markdown(rows or "<span class='subtle'>Keine Merkmale gesetzt.</span>", unsafe_allow_html=True)
    return order, proba


def render_results():
    st.markdown("## Risk Assessment Results")
    st.markdown(f"<div class='subtle'>{ss.portfolio_name}</div>", unsafe_allow_html=True)
    if not ss.projects:
        st.info("Noch keine Projekte. Wechsle zu **Configure**, fuege ein Projekt hinzu und berechne.")
        return

    per_project = [mb.predict(MODEL, META, p["params"]) for p in ss.projects]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]

    st.markdown(
        f"""<div class="section-card" style="margin-bottom:1.4rem;">
            <div style="font-size:1.2rem; font-weight:600; margin-bottom:1.1rem;">Portfolio Risk Summary</div>
            <div style="display:flex; gap:3rem;">
                <div><div class="subtle">Total Portfolio Risk</div>
                    <div style="font-size:2rem; font-weight:700; color:{p_color};">{pm['level']}</div></div>
                <div style="margin-left:auto; text-align:right;"><div class="subtle">P(&#8805;1 elevated-risk project)</div>
                    <div style="font-size:1.6rem; font-weight:600;">{pm['p_at_least_one_elevated']:.0%}</div></div>
                <div style="text-align:right;"><div class="subtle">Expected # &#8805; High</div>
                    <div style="font-size:1.6rem; font-weight:600;">{pm['expected_elevated_count']:.1f}</div></div>
                <div style="text-align:right;"><div class="subtle">Projects</div>
                    <div style="font-size:1.6rem; font-weight:600;">{pm['n']}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

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
