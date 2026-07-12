"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import streamlit as st
import model_bridge as mb

# --------------------------------------------------------------------------------------
# Design-Tokens (Dark; Weiss nur als Textfarbe, nie als Flaeche)
# --------------------------------------------------------------------------------------
BG      = "#1b1917"
PANEL   = "#231f1d"
PANEL_2 = "#2a2523"
BORDER  = "#3a342f"
TEXT    = "#e6e3e0"
MUTED   = "#8a8178"
ACCENT  = "#a8a29a"
GREEN   = "#4a9d5b"
RED     = "#e5484d"

LEVEL_COLORS = {"Low": GREEN, "Medium": "#d9a441", "High": "#e5844d", "Critical": RED, "N/A": MUTED}
AGG_LEVELS = ["N/A", "Low", "Medium", "High", "Very High"]

# --------------------------------------------------------------------------------------
# Artefakte + SHAP-Explainer
# --------------------------------------------------------------------------------------
@st.cache_resource
def _load():
    model, meta, spec, df = mb.load_all()
    return model, meta, spec, mb.build_explainer(model, df)

try:
    MODEL, META, SPEC, CTX = _load()
    LOAD_ERROR = None
except Exception as e:
    MODEL = META = SPEC = CTX = None
    LOAD_ERROR = str(e)

# --------------------------------------------------------------------------------------
# Setup + CSS (Widgets explizit dunkel, damit keine weissen Kaesten entstehen)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Assessment",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color:{BG}; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
    .block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1050px; }}
    html, body, [class*="css"], p, label, span {{ color:{TEXT}; }}
    h1, h2, h3 {{ font-weight:600; letter-spacing:-0.01em; color:{TEXT}; }}
    .subtle {{ color:{MUTED}; font-size:0.95rem; }}
    .cat-header {{ color:{ACCENT}; text-transform:uppercase; letter-spacing:0.12em;
                   font-size:0.8rem; font-weight:700; }}
    .param-label {{ font-weight:600; font-size:0.88rem; margin:0.5rem 0 0.1rem 0; color:{TEXT}; }}
    .param-label span, .info span {{ cursor:help; }}
    .tick-row {{ display:flex; justify-content:space-between; color:{MUTED};
                 font-size:0.68rem; margin:-0.3rem 0 0.15rem 0; }}
    .divider {{ height:1px; background:{BORDER}; margin:1.1rem 0; border:none; }}

    /* Eingaben dunkel */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
        background-color:{PANEL_2} !important; color:{TEXT} !important;
        border:1px solid {BORDER} !important; border-radius:8px !important; }}
    [data-baseweb="select"] > div {{
        background-color:{PANEL_2} !important; border-color:{BORDER} !important; color:{TEXT} !important; }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{ background-color:{PANEL} !important; }}

    /* Buttons dunkel */
    .stButton > button {{ background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
                          border-radius:8px; font-weight:500; padding:0.35rem 0.8rem; }}
    .stButton > button:hover {{ border-color:{ACCENT}; color:{TEXT}; }}
    div[role="radiogroup"] {{ gap:0.3rem; }}
</style>
""", unsafe_allow_html=True)

if LOAD_ERROR:
    st.error("Modell-Artefakte konnten nicht geladen werden. Bitte zuerst `masterskript_final.py` "
             f"lokal ausfuehren.\n\nDetails: {LOAD_ERROR}")
    st.stop()

# --------------------------------------------------------------------------------------
# Session-State
# --------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("portfolios", {})
ss.setdefault("active", None)
ss.setdefault("view", "Configure")
ss.setdefault("draft_id", 0)
ss.setdefault("pf_counter", 0)
MIN_FEATURES = 5


def new_portfolio():
    ss.pf_counter += 1
    pid = f"pf{ss.pf_counter}"
    ss.portfolios[pid] = {"name": f"Portfolio {ss.pf_counter}", "projects": []}
    ss.active = pid
    ss.view = "Configure"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Portfolios")
    if st.button("\uFF0B  New Portfolio", use_container_width=True):
        new_portfolio(); st.rerun()
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if ss.portfolios:
        for pid, pf in ss.portfolios.items():
            mark = "\u25CF " if pid == ss.active else "\u25CB "
            if st.button(mark + pf["name"], key=f"sel_{pid}", use_container_width=True):
                ss.active = pid; st.rerun()
    else:
        st.caption("No portfolios yet.")


# --------------------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------------------
def _ticks(opts):
    return "<div class='tick-row'>" + "".join(f"<span>{o}</span>" for o in opts) + "</div>"


def _dir_bar(feat):
    d = SPEC[feat]["direction"]
    if d == 0:
        return "<div style='margin-bottom:0.55rem;'></div>"
    grad = f"to right, {GREEN}, {RED}" if d > 0 else f"to right, {RED}, {GREEN}"
    return (f"<div style='display:flex; justify-content:center; margin:0.05rem 0 0.55rem 0;'>"
            f"<div title='Farbverlauf zeigt die Risikorichtung des Sliders' "
            f"style='width:55%; height:5px; border-radius:3px; background:linear-gradient({grad});'></div></div>")


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
            new_portfolio(); st.rerun()


# ======================================================================================
# CONFIGURE
# ======================================================================================
def render_feature(pid, feat):
    spec = SPEC[feat]
    opts = spec["options"]
    st.markdown(f"<div class='param-label'><span title='{spec['tooltip']}'>{spec['label']} &#9432;</span></div>",
                unsafe_allow_html=True)
    slider_opts = ["N/A"] + list(opts)

    if spec["type"] == "nominal":
        choice = st.selectbox(" ", slider_opts, key=f"in_{pid}_{feat}", label_visibility="collapsed")
        st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
        return None if choice == "N/A" else spec["value_map"][choice]

    if spec["type"] == "numeric":
        c1, c2 = st.columns([0.76, 0.24])
        with c1:
            choice = st.select_slider(" ", options=slider_opts, value="N/A",
                                      key=f"in_{pid}_{feat}", label_visibility="collapsed")
        with c2:
            raw = st.text_input(" ", key=f"num_{pid}_{feat}", placeholder="custom",
                                label_visibility="collapsed")
        st.markdown(_ticks(slider_opts) + _dir_bar(feat), unsafe_allow_html=True)
        if raw:
            try:
                return float(raw.replace(",", "."))
            except ValueError:
                pass
        return None if choice == "N/A" else spec["value_map"][choice]

    choice = st.select_slider(" ", options=slider_opts, value="N/A",
                              key=f"in_{pid}_{feat}", label_visibility="collapsed")
    st.markdown(_ticks(slider_opts) + _dir_bar(feat), unsafe_allow_html=True)
    return None if choice == "N/A" else spec["value_map"][choice]


def render_category_aggregate(pid, crit, feats):
    st.markdown(f"<div class='param-label'><span title='Setzt alle Merkmale dieser Kategorie auf die "
                f"gewaehlte Stufe.'>Overall {crit} &#9432;</span></div>", unsafe_allow_html=True)
    choice = st.select_slider(" ", options=AGG_LEVELS, value="N/A",
                              key=f"agg_{pid}_{crit}", label_visibility="collapsed")
    st.markdown(_ticks(AGG_LEVELS) + "<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    if choice == "N/A":
        return {f: None for f in feats}
    frac = (AGG_LEVELS.index(choice) - 1) / (len(AGG_LEVELS) - 2)
    return {f: SPEC[f]["value_map"][SPEC[f]["options"][round(frac * (len(SPEC[f]["options"]) - 1))]]
            for f in feats}


def render_configure(pf):
    st.markdown("## Portfolio Configuration")
    pf["name"] = st.text_input("Portfolio Name", value=pf["name"])

    if pf["projects"]:
        st.markdown("<span class='subtle'>Added Projects</span>", unsafe_allow_html=True)
        for i, proj in enumerate(pf["projects"]):
            row_l, row_r = st.columns([0.88, 0.12])
            row_l.markdown(f"<div style='padding:0.35rem 0;'>{proj['name']}</div>", unsafe_allow_html=True)
            if row_r.button("Remove", key=f"del_{i}"):
                pf["projects"].pop(i); st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pid = ss.draft_id
    with st.container(border=True):
        proj_name = st.text_input("Project Name", value=f"Project {len(pf['projects']) + 1}", key=f"pname_{pid}")
        st.caption(f"Mindestens {MIN_FEATURES} Merkmale angeben. Je mehr Merkmale gesetzt sind, "
                   f"desto zuverlaessiger die Vorhersage.")

        draft = {}
        for crit, feats in mb.KUB_GROUPS.items():
            with st.container(border=True):
                h1, h2 = st.columns([0.6, 0.4])
                h1.markdown(f"<div class='cat-header' style='padding-top:0.4rem;'>{crit}</div>",
                            unsafe_allow_html=True)
                with h2:
                    detailed = st.toggle("Einzeln einstellen", key=f"tg_{pid}_{crit}")
                if detailed:
                    for f in feats:
                        draft[f] = render_feature(pid, f)
                else:
                    draft.update(render_category_aggregate(pid, crit, feats))

    n_set = sum(v is not None for v in draft.values())
    _, a_col, b_col, _ = st.columns([0.28, 0.22, 0.22, 0.28])
    with a_col:
        if st.button(f"Add Project ({n_set})", use_container_width=True):
            if n_set < MIN_FEATURES:
                st.warning(f"Bitte mindestens {MIN_FEATURES} Merkmale setzen (aktuell {n_set}).")
            else:
                pf["projects"].append({"name": proj_name, "params": draft})
                ss.draft_id += 1; st.rerun()
    with b_col:
        if st.button("Calculate Results", use_container_width=True):
            if not pf["projects"]:
                st.warning("Bitte zuerst mindestens ein Projekt hinzufuegen.")
            else:
                ss.view = "Results"; st.rerun()


# ======================================================================================
# RESULTS
# ======================================================================================
def prob_bars(order, proba, pred):
    rows = ""
    for cls, p in zip(order, proba):
        c = LEVEL_COLORS[cls]
        strong = "font-weight:700;" if cls == pred else "opacity:0.8;"
        rows += (f"<div style='display:flex; align-items:center; gap:0.8rem; margin:0.25rem 0;'>"
                 f"<div style='width:70px; color:{MUTED}; font-size:0.85rem;'>{cls}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:6px; height:14px;'>"
                 f"<div style='width:{p*100:.1f}%; background:{c}; height:14px; border-radius:6px;'></div></div>"
                 f"<div style='width:52px; text-align:right; color:{c}; {strong} font-size:0.85rem;'>{p:.1%}</div>"
                 f"</div>")
    return rows


def shap_drivers(order, proba, params):
    drivers = mb.explain(CTX, META, params, order, proba, top_n=5)
    maxabs = max((abs(v) for _, v, _ in drivers), default=1) or 1
    rows = ""
    for feat, v, is_set in drivers:
        c = RED if v > 0 else GREEN
        tag = "" if is_set else f" <span style='color:{MUTED}; font-size:0.72rem;'>(default)</span>"
        rows += (f"<div style='display:flex; align-items:center; gap:0.7rem; margin:0.22rem 0;'>"
                 f"<div style='width:190px; font-size:0.82rem;'>{mb.label_of(feat)}{tag}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:5px; height:10px;'>"
                 f"<div style='width:{abs(v)/maxabs*100:.0f}%; background:{c}; height:10px; border-radius:5px;'></div></div>"
                 f"<div style='width:56px; text-align:right; color:{c}; font-size:0.8rem; font-weight:600;'>{v:+.2f}</div>"
                 f"</div>")
    return ("<div class='subtle' style='font-size:0.75rem; margin-bottom:0.3rem;'>"
            "SHAP-Beitrag zur vorhergesagten Klasse (+ erh&ouml;ht, &minus; senkt)</div>") + rows


def render_project_card(proj):
    order, proba = mb.predict(MODEL, META, proj["params"])
    pred = order[list(proba).index(max(proba))]
    color = LEVEL_COLORS[pred]
    score = mb.expected_score(order, proba)
    n_set = sum(v is not None for v in proj["params"].values())
    score_tip = ("Erwartungswert der Risikostufe (Low=1 .. Critical=4), gewichtet mit den "
                 "Klassenwahrscheinlichkeiten.")

    with st.container(border=True):
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:1.1rem; font-weight:600;">{proj['name']}</div>
                <div style="color:{color}; font-weight:700;">{pred}</div></div>
            <div class="info subtle" style="margin:0.1rem 0 0.7rem 0;">
                <span title="{score_tip}">Expected score: {score:.2f} &#9432;</span>
                &middot; {n_set} feature(s) set</div>
            {prob_bars(order, proba, pred)}""", unsafe_allow_html=True)
        with st.expander("Top drivers & parameters"):
            st.markdown("<div class='cat-header'>TOP DRIVERS (SHAP)</div>", unsafe_allow_html=True)
            st.markdown(shap_drivers(order, proba, proj["params"]), unsafe_allow_html=True)
            st.markdown("<div class='cat-header' style='margin-top:1rem;'>SET PARAMETERS</div>",
                        unsafe_allow_html=True)
            rows = ""
            for feat, v in proj["params"].items():
                if v is not None:
                    val = round(v, 2) if isinstance(v, float) else v
                    rows += (f"<div style='display:flex; justify-content:space-between; padding:0.2rem 0;"
                             f" border-bottom:1px solid {BORDER};'>"
                             f"<span style='color:{MUTED};'>{mb.label_of(feat)}</span>"
                             f"<span style='font-weight:600;'>{val}</span></div>")
            st.markdown(rows or "<span class='subtle'>Keine Merkmale gesetzt.</span>", unsafe_allow_html=True)


def render_results(pf):
    st.markdown("## Risk Assessment Results")
    st.markdown(f"<div class='subtle'>{pf['name']}</div>", unsafe_allow_html=True)
    if not pf["projects"]:
        st.info("Noch keine Projekte. Wechsle zu **Configure**, fuege ein Projekt hinzu und berechne.")
        return

    per_project = [mb.predict(MODEL, META, p["params"]) for p in pf["projects"]]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]
    tip_elev = ("Wahrscheinlichkeit, dass mindestens ein Projekt im Portfolio High- oder Critical-Risiko "
                "hat. Annahme: Projekte unabhaengig (Baumdiagramm-Pfad).")
    tip_cnt = "Erwartete Anzahl Projekte mit High- oder Critical-Risiko (Summe der Einzelwahrscheinlichkeiten)."

    with st.container(border=True):
        st.markdown(
            f"""<div style="font-size:1.1rem; font-weight:600; margin-bottom:1rem;">Portfolio Risk Summary</div>
            <div class="info" style="display:flex; gap:2.5rem; flex-wrap:wrap;">
                <div><div class="subtle">Total Portfolio Risk</div>
                    <div style="font-size:1.9rem; font-weight:700; color:{p_color};">{pm['level']}</div></div>
                <div style="margin-left:auto; text-align:right;">
                    <div class="subtle"><span title="{tip_elev}">P(&#8805;1 elevated-risk project) &#9432;</span></div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['p_at_least_one_elevated']:.0%}</div></div>
                <div style="text-align:right;">
                    <div class="subtle"><span title="{tip_cnt}">Expected # &#8805; High &#9432;</span></div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['expected_elevated_count']:.1f}</div></div>
                <div style="text-align:right;"><div class="subtle">Projects</div>
                    <div style="font-size:1.5rem; font-weight:600;">{pm['n']}</div></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Project Risk Breakdown")
    for proj in pf["projects"]:
        render_project_card(proj)


# ======================================================================================
# ROUTER
# ======================================================================================
nav_col, _ = st.columns([0.35, 0.65])
with nav_col:
    ss.view = st.radio("nav", ["Configure", "Results"],
                       index=0 if ss.view == "Configure" else 1,
                       horizontal=True, label_visibility="collapsed")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if ss.active is None or ss.active not in ss.portfolios:
    render_empty_state()
elif ss.view == "Configure":
    render_configure(ss.portfolios[ss.active])
else:
    render_results(ss.portfolios[ss.active])
