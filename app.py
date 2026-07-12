"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import streamlit as st
import model_bridge as mb

# --------------------------------------------------------------------------------------
# Design-Tokens (helle, Claude-nahe Palette: viel Grau/Weiss, Farbe nur fuer Risiko)
# --------------------------------------------------------------------------------------
BG      = "#faf9f5"
PANEL   = "#ffffff"
PANEL_2 = "#f0eee6"
BORDER  = "#e3e1d9"
TEXT    = "#1f1e1d"
MUTED   = "#73716b"
ACCENT  = "#6b6a65"

LEVEL_COLORS = {
    "Low":      "#4a9d5b",
    "Medium":   "#c99a3a",
    "High":     "#d0743c",
    "Critical": "#c0453f",
    "N/A":      MUTED,
}
AGG_LEVELS = ["N/A", "Low", "Medium", "High", "Very High"]

# --------------------------------------------------------------------------------------
# Artefakte laden
# --------------------------------------------------------------------------------------
@st.cache_resource
def _load():
    return mb.load_all()

try:
    MODEL, META, SPEC = _load()
    LOAD_ERROR = None
except Exception as e:
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
    .block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1100px; }}
    html, body, [class*="css"] {{ color:{TEXT}; }}
    h1, h2, h3 {{ font-weight:600; letter-spacing:-0.01em; color:{TEXT}; }}
    .subtle {{ color:{MUTED}; font-size:0.95rem; }}
    .cat-header {{
        color:{ACCENT}; text-transform:uppercase; letter-spacing:0.12em;
        font-size:0.82rem; font-weight:700; margin:0.35rem 0;
    }}
    .param-label {{ font-weight:600; font-size:0.9rem; margin:0.5rem 0 0.15rem 0; color:{TEXT}; }}
    .param-label span {{ cursor:help; }}
    .tick-row {{
        display:flex; justify-content:space-between; color:{MUTED};
        font-size:0.7rem; margin:-0.35rem 0 0.6rem 0;
    }}
    .divider {{ height:1px; background:{BORDER}; margin:1.2rem 0; border:none; }}
    .stButton > button {{
        background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
        border-radius:8px; font-weight:500; padding:0.4rem 0.9rem;
    }}
    .stButton > button:hover {{ border-color:{ACCENT}; color:{TEXT}; }}
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
ss.setdefault("projects", [])
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

# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
nav_col, _ = st.columns([0.35, 0.65])
with nav_col:
    ss.view = st.radio("nav", ["Configure", "Results"],
                       index=0 if ss.view == "Configure" else 1,
                       horizontal=True, label_visibility="collapsed")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)


def _tooltip(opts):
    return "Stufen (links -&gt; rechts): " + ", ".join(str(o) for o in opts)


def _ticks(opts):
    return "<div class='tick-row'>" + "".join(f"<span>{o}</span>" for o in opts) + "</div>"


# ======================================================================================
# EMPTY STATE
# ======================================================================================
def render_empty_state():
    st.markdown(
        f"""<div style="text-align:center; padding:6rem 0;">
            <div style="font-size:1.4rem; font-weight:600;">No Portfolio Selected</div>
            <div class="subtle">Create a new portfolio to get started</div></div>""",
        unsafe_allow_html=True)
    _, c2, _ = st.columns([1, 1, 1])
    with c2:
        if st.button("Create New Portfolio", use_container_width=True):
            ss.portfolio_created = True
            ss.view = "Configure"
            st.rerun()


# ======================================================================================
# CONFIGURE
# ======================================================================================
def render_feature(pid, feat):
    spec = SPEC[feat]
    opts = spec["options"]
    st.markdown(f"<div class='param-label'><span title='{_tooltip(opts)}'>{feat.replace('_',' ')} &#9432;</span></div>",
                unsafe_allow_html=True)
    slider_opts = ["N/A"] + list(opts)
    if spec["type"] == "nominal":
        choice = st.selectbox(" ", slider_opts, key=f"in_{pid}_{feat}", label_visibility="collapsed")
    else:
        choice = st.select_slider(" ", options=slider_opts, value="N/A",
                                  key=f"in_{pid}_{feat}", label_visibility="collapsed")
        st.markdown(_ticks(slider_opts), unsafe_allow_html=True)
    return None if choice == "N/A" else spec["value_map"][choice]


def render_category_aggregate(pid, crit, feats):
    st.markdown(f"<div class='param-label'><span title='Setzt alle Merkmale dieser Kategorie auf die "
                f"gewaehlte Stufe.'>Overall {crit} &#9432;</span></div>", unsafe_allow_html=True)
    choice = st.select_slider(" ", options=AGG_LEVELS, value="N/A",
                              key=f"agg_{pid}_{crit}", label_visibility="collapsed")
    st.markdown(_ticks(AGG_LEVELS), unsafe_allow_html=True)
    if choice == "N/A":
        return {f: None for f in feats}
    frac = (AGG_LEVELS.index(choice) - 1) / (len(AGG_LEVELS) - 2)
    out = {}
    for f in feats:
        o = SPEC[f]["options"]
        out[f] = SPEC[f]["value_map"][o[round(frac * (len(o) - 1))]]
    return out


def render_configure():
    st.markdown("## Portfolio Configuration")
    ss.portfolio_name = st.text_input("Portfolio Name", value=ss.portfolio_name)

    if ss.projects:
        st.markdown("<span class='subtle'>Added Projects</span>", unsafe_allow_html=True)
        for i, proj in enumerate(ss.projects):
            row_l, row_r = st.columns([0.9, 0.1])
            row_l.markdown(f"<div style='padding:0.35rem 0;'>{proj['name']}</div>", unsafe_allow_html=True)
            if row_r.button("Remove", key=f"del_{i}"):
                ss.projects.pop(i)
                st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pid = ss.draft_id
    with st.container(border=True):
        proj_name = st.text_input("Project Name", value=f"Project {len(ss.projects) + 1}", key=f"pname_{pid}")
        st.caption(f"Mindestens {MIN_FEATURES} Merkmale angeben. Je mehr Merkmale gesetzt sind, "
                   f"desto zuverlaessiger die Vorhersage. Klappe eine Kategorie fuer die "
                   f"Einzeleinstellung ihrer Merkmale auf.")

        draft = {}
        for crit, feats in mb.KUB_GROUPS.items():
            with st.container(border=True):
                h1, h2 = st.columns([0.55, 0.45])
                h1.markdown(f"<div class='cat-header'>{crit}</div>", unsafe_allow_html=True)
                detailed = h2.toggle("Einzeln einstellen", key=f"tg_{pid}_{crit}")
                if detailed:
                    for f in feats:
                        draft[f] = render_feature(pid, f)
                else:
                    draft.update(render_category_aggregate(pid, crit, feats))

    n_set = sum(v is not None for v in draft.values())
    a_col, b_col, _ = st.columns([0.26, 0.26, 0.48])
    with a_col:
        if st.button(f"Add Project ({n_set} set)", use_container_width=True):
            if n_set < MIN_FEATURES:
                st.warning(f"Bitte mindestens {MIN_FEATURES} Merkmale setzen (aktuell {n_set}).")
            else:
                ss.projects.append({"name": proj_name, "params": draft})
                ss.draft_id += 1
                st.rerun()
    with b_col:
        if st.button("Calculate Results", use_container_width=True):
            if not ss.projects:
                st.warning("Bitte zuerst mindestens ein Projekt hinzufuegen.")
            else:
                ss.view = "Results"
                st.rerun()


# ======================================================================================
# RESULTS
# ======================================================================================
def prob_bars(order, proba, pred):
    rows = ""
    for cls, p in zip(order, proba):
        c = LEVEL_COLORS[cls]
        strong = "font-weight:700;" if cls == pred else "opacity:0.8;"
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

    with st.container(border=True):
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:1.1rem; font-weight:600;">{proj['name']}</div>
                <div style="color:{color}; font-weight:700;">{pred}</div></div>
            <div style="color:{MUTED}; font-size:0.85rem; margin:0.1rem 0 0.7rem 0;">
                Expected score: {score:.2f} &middot; {n_set} feature(s) set</div>
            {prob_bars(order, proba, pred)}""", unsafe_allow_html=True)
        with st.expander("Show set parameters"):
            rows = ""
            for feat, v in proj["params"].items():
                if v is not None:
                    val = round(v, 2) if isinstance(v, float) else v
                    rows += (f"<div style='display:flex; justify-content:space-between; padding:0.25rem 0;"
                             f" border-bottom:1px solid {BORDER};'>"
                             f"<span style='color:{MUTED};'>{feat.replace('_',' ')}</span>"
                             f"<span style='font-weight:600;'>{val}</span></div>")
            st.markdown(rows or "<span class='subtle'>Keine Merkmale gesetzt.</span>", unsafe_allow_html=True)


def render_results():
    st.markdown("## Risk Assessment Results")
    st.markdown(f"<div class='subtle'>{ss.portfolio_name}</div>", unsafe_allow_html=True)
    if not ss.projects:
        st.info("Noch keine Projekte. Wechsle zu **Configure**, fuege ein Projekt hinzu und berechne.")
        return

    per_project = [mb.predict(MODEL, META, p["params"]) for p in ss.projects]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]

    with st.container(border=True):
        st.markdown(
            f"""<div style="font-size:1.1rem; font-weight:600; margin-bottom:1rem;">Portfolio Risk Summary</div>
            <div style="display:flex; gap:2.5rem; flex-wrap:wrap;">
                <div><div class="subtle">Total Portfolio Risk</div>
                    <div style="font-size:1.9rem; font-weight:700; color:{p_color};">{pm['level']}</div></div>
                <div style="margin-left:auto; text-align:right;"><div class="subtle">P(&#8805;1 elevated-risk project)</div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['p_at_least_one_elevated']:.0%}</div></div>
                <div style="text-align:right;"><div class="subtle">Expected # &#8805; High</div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['expected_elevated_count']:.1f}</div></div>
                <div style="text-align:right;"><div class="subtle">Projects</div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['n']}</div></div>
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
