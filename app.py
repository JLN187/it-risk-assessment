"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import streamlit as st
import model_bridge as mb

# --------------------------------------------------------------------------------------
# Design-Tokens (neutrales Dark-Grau, klarer Kontrast; Weiss nur als Textfarbe)
# --------------------------------------------------------------------------------------
BG      = "#262624"
PANEL   = "#31312e"
PANEL_2 = "#3c3c38"
BORDER  = "#55554f"
TEXT    = "#f2f0ea"
MUTED   = "#b3b0a8"
HEAD    = "#d8d5cd"
GREEN   = "#5bb56b"
RED     = "#e0574f"

LEVEL_COLORS = {"Low": GREEN, "Medium": "#d9a441", "High": "#e5844d", "Critical": RED, "N/A": MUTED}
AGG_LEVELS = ["N/A", "Low", "Medium", "High", "Very High"]

# --------------------------------------------------------------------------------------
# i18n
# --------------------------------------------------------------------------------------
STR = {
    "en": {
        "configure": "Configure", "results": "Results",
        "portfolios": "Portfolios", "new_portfolio": "New Portfolio", "no_portfolios": "No portfolios yet.",
        "no_pf_sel": "No Portfolio Selected", "create_pf": "Create a new portfolio to get started",
        "create_btn": "Create New Portfolio", "language": "Language",
        "pf_config": "Portfolio Configuration", "pf_name": "Portfolio Name",
        "added_projects": "Added Projects", "remove": "Remove", "project_name": "Project Name",
        "min_hint": "Set at least {n} features. The more features you set, the more reliable the prediction.",
        "single": "Set individually", "overall": "Overall",
        "add_project": "Add Project", "calc": "Calculate Results",
        "warn_min": "Please set at least {n} features (currently {c}).",
        "warn_add": "Please add at least one project first.",
        "risk_results": "Risk Assessment Results", "pf_summary": "Portfolio Risk Summary",
        "total_risk": "Total Portfolio Risk", "p_elev": "P(\u22651 elevated-risk project)",
        "exp_high": "Expected # \u2265 High", "projects": "Projects", "breakdown": "Project Risk Breakdown",
        "no_projects": "No projects yet. Go to Configure, add a project and calculate.",
        "expander": "Top drivers & parameters", "top_drivers": "TOP DRIVERS (SHAP)",
        "shap_note": "SHAP contribution to the predicted class (+ raises, \u2212 lowers)",
        "set_params": "SET PARAMETERS", "no_set": "No features set.",
        "exp_score": "Expected score", "features_set": "feature(s) set", "custom": "custom",
        "levels": "Levels (left \u2192 right): ",
        "tip_score": "Expected value of the risk level (Low=1 .. Critical=4), weighted by class probabilities.",
        "tip_elev": "Probability that at least one project in the portfolio is High or Critical risk. Assumes projects are independent (probability-tree path).",
        "tip_cnt": "Expected number of projects at High or Critical risk (sum of individual probabilities).",
        "tip_dir": "Colour gradient shows the risk direction of the slider (green = lower risk, red = higher risk).",
        "tip_agg": "Sets all features of this category to the chosen level.",
        "load_err": "Model artifacts could not be loaded. Please run masterskript_final.py locally first.",
    },
    "de": {
        "configure": "Konfigurieren", "results": "Ergebnisse",
        "portfolios": "Portfolios", "new_portfolio": "Neues Portfolio", "no_portfolios": "Noch keine Portfolios.",
        "no_pf_sel": "Kein Portfolio ausgew\u00e4hlt", "create_pf": "Erstelle ein Portfolio, um zu starten",
        "create_btn": "Neues Portfolio erstellen", "language": "Sprache",
        "pf_config": "Portfolio-Konfiguration", "pf_name": "Portfolioname",
        "added_projects": "Hinzugef\u00fcgte Projekte", "remove": "Entfernen", "project_name": "Projektname",
        "min_hint": "Mindestens {n} Merkmale angeben. Je mehr Merkmale gesetzt sind, desto zuverl\u00e4ssiger die Vorhersage.",
        "single": "Einzeln einstellen", "overall": "Gesamt",
        "add_project": "Projekt hinzuf\u00fcgen", "calc": "Ergebnisse berechnen",
        "warn_min": "Bitte mindestens {n} Merkmale setzen (aktuell {c}).",
        "warn_add": "Bitte zuerst mindestens ein Projekt hinzuf\u00fcgen.",
        "risk_results": "Risikobewertung", "pf_summary": "Portfolio-Risiko\u00fcbersicht",
        "total_risk": "Gesamt-Portfoliorisiko", "p_elev": "P(\u22651 Hochrisikoprojekt)",
        "exp_high": "Erwartete Anzahl \u2265 High", "projects": "Projekte", "breakdown": "Projekt-Risiko im Detail",
        "no_projects": "Noch keine Projekte. Wechsle zu Konfigurieren, f\u00fcge ein Projekt hinzu und berechne.",
        "expander": "Top-Treiber & Parameter", "top_drivers": "TOP-TREIBER (SHAP)",
        "shap_note": "SHAP-Beitrag zur vorhergesagten Klasse (+ erh\u00f6ht, \u2212 senkt)",
        "set_params": "GESETZTE MERKMALE", "no_set": "Keine Merkmale gesetzt.",
        "exp_score": "Erwartungswert-Score", "features_set": "Merkmal(e) gesetzt", "custom": "eigener Wert",
        "levels": "Stufen (links \u2192 rechts): ",
        "tip_score": "Erwartungswert der Risikostufe (Low=1 .. Critical=4), gewichtet mit den Klassenwahrscheinlichkeiten.",
        "tip_elev": "Wahrscheinlichkeit, dass mindestens ein Projekt High- oder Critical-Risiko hat. Annahme: Projekte unabh\u00e4ngig (Baumdiagramm-Pfad).",
        "tip_cnt": "Erwartete Anzahl Projekte mit High- oder Critical-Risiko (Summe der Einzelwahrscheinlichkeiten).",
        "tip_dir": "Farbverlauf zeigt die Risikorichtung des Sliders (gr\u00fcn = weniger Risiko, rot = mehr Risiko).",
        "tip_agg": "Setzt alle Merkmale dieser Kategorie auf die gew\u00e4hlte Stufe.",
        "load_err": "Modell-Artefakte konnten nicht geladen werden. Bitte zuerst masterskript_final.py lokal ausf\u00fchren.",
    },
}
SEMANTIC = {
    "Project_Start_Month": {"en": "Calendar month the project starts (1=Jan..12=Dec); captures seasonal effects.",
                            "de": "Kalendermonat des Projektstarts (1=Jan..12=Dez); bildet saisonale Effekte ab."},
    "Seasonal_Risk_Factor": {"en": "Seasonal risk factor of the project (~1.0-1.1).",
                             "de": "Saisonaler Risikofaktor (ca. 1.0-1.1)."},
    "Budget_Utilization_Rate": {"en": "Share of budget expected to be consumed (can exceed 100%).",
                                "de": "Anteil des Budgets, der voraussichtlich verbraucht wird (kann >100% sein)."},
    "Resource_Availability": {"en": "Availability of required resources (0-100%).",
                              "de": "Verf\u00fcgbarkeit ben\u00f6tigter Ressourcen (0-100%)."},
    "Technical_Debt_Level": {"en": "Level of accumulated technical debt (0-100%).",
                             "de": "Grad der technischen Schulden (0-100%)."},
    "Team_Turnover_Rate": {"en": "Expected team turnover (0-100%).",
                           "de": "Erwartete Personalfluktuation (0-100%)."},
    "Vendor_Reliability_Score": {"en": "Reliability of external vendors (0-100%).",
                                 "de": "Zuverl\u00e4ssigkeit externer Dienstleister (0-100%)."},
}

# --------------------------------------------------------------------------------------
# Laden
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
# Setup + CSS
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Risk Assessment", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color:{BG}; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
    .block-container {{ padding-top:2rem; padding-bottom:4rem; max-width:1050px; }}
    html, body, p, label, span, div {{ color:{TEXT}; }}
    h1, h2, h3 {{ font-weight:600; letter-spacing:-0.01em; color:{TEXT} !important; }}
    .subtle {{ color:{MUTED}; font-size:0.95rem; }}
    .cat-header {{ color:{HEAD}; text-transform:uppercase; letter-spacing:0.12em;
                   font-size:0.8rem; font-weight:700; }}
    .param-label {{ font-weight:600; font-size:0.88rem; margin:0.5rem 0 0.1rem 0; color:{TEXT}; }}
    .param-label span, .info span {{ cursor:help; }}
    .tick-row {{ display:flex; justify-content:space-between; color:{MUTED};
                 font-size:0.68rem; margin:-0.3rem 0 0.55rem 0; }}
    .divider {{ height:1px; background:{BORDER}; margin:1rem 0; border:none; }}

    /* Eingaben & Auswahl dunkel */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
        background-color:{PANEL_2} !important; color:{TEXT} !important;
        border:1px solid {BORDER} !important; border-radius:8px !important; }}
    [data-baseweb="select"] > div {{ background-color:{PANEL_2} !important;
        border-color:{BORDER} !important; color:{TEXT} !important; }}
    [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {{
        background-color:{PANEL} !important; }}
    /* Slider-Griff neutral (kein Orange) */
    [data-baseweb="slider"] div[role="slider"] {{ background-color:{HEAD} !important; }}

    /* Bordered Container + Expander dunkel, klare Rahmen */
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER} !important; border-radius:10px; }}
    [data-testid="stExpander"] details {{ background:{PANEL} !important;
        border:1px solid {BORDER} !important; border-radius:8px !important; }}
    [data-testid="stExpander"] summary {{ background:{PANEL} !important; color:{TEXT} !important; }}
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {{ color:{TEXT} !important; }}

    /* Buttons */
    .stButton > button {{ background:{PANEL_2}; color:{TEXT}; border:1px solid {BORDER};
        border-radius:8px; font-weight:500; padding:0.35rem 0.8rem; }}
    .stButton > button:hover {{ border-color:{HEAD}; color:{TEXT}; }}
    div[role="radiogroup"] {{ gap:0.3rem; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Session-State + Sprache
# --------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("lang", "en")
ss.setdefault("portfolios", {})
ss.setdefault("active", None)
ss.setdefault("view", "Configure")
ss.setdefault("draft_id", 0)
ss.setdefault("pf_counter", 0)
MIN_FEATURES = 5


def T(k, **kw):
    s = STR[ss.lang].get(k, k)
    return s.format(**kw) if kw else s


if LOAD_ERROR:
    st.error(T("load_err") + f"\n\nDetails: {LOAD_ERROR}")
    st.stop()


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
    lang_choice = st.radio(T("language"), ["EN", "DE"], horizontal=True,
                           index=0 if ss.lang == "en" else 1)
    ss.lang = "en" if lang_choice == "EN" else "de"
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"### {T('portfolios')}")
    if st.button("\uFF0B  " + T("new_portfolio"), use_container_width=True):
        new_portfolio(); st.rerun()
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if ss.portfolios:
        for pid, pf in ss.portfolios.items():
            mark = "\u25CF " if pid == ss.active else "\u25CB "
            if st.button(mark + pf["name"], key=f"sel_{pid}", use_container_width=True):
                ss.active = pid; st.rerun()
    else:
        st.caption(T("no_portfolios"))


# --------------------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------------------
def _ticks(opts):
    return "<div class='tick-row'>" + "".join(f"<span>{o}</span>" for o in opts) + "</div>"


def _mini(direction):
    """Kleiner Verlaufs-Indikator neben dem Label (Risikorichtung)."""
    if direction == 0:
        return ""
    grad = f"to right,{GREEN},{RED}" if direction > 0 else f"to right,{RED},{GREEN}"
    return (f"<span title='{T('tip_dir')}' style='display:inline-block; width:26px; height:6px; "
            f"border-radius:3px; vertical-align:middle; margin-left:8px; "
            f"background:linear-gradient({grad});'></span>")


def _tooltip(feat):
    sem = SEMANTIC.get(feat, {}).get(ss.lang, "")
    lv = T("levels") + ", ".join(str(o) for o in SPEC[feat]["options"])
    return (sem + " " + lv).strip()


def _cat_dir(feats):
    s = sum(SPEC[f]["direction"] for f in feats)
    return 1 if s > 0 else (-1 if s < 0 else 0)


def _label(feat):
    return (f"<div class='param-label'><span title='{_tooltip(feat)}'>{mb.label_of(feat)} &#9432;</span>"
            f"{_mini(SPEC[feat]['direction'])}</div>")


# ======================================================================================
# EMPTY STATE
# ======================================================================================
def render_empty_state():
    st.markdown(
        f"""<div style="text-align:center; padding:6rem 0;">
            <div style="font-size:1.4rem; font-weight:600; color:{TEXT};">{T('no_pf_sel')}</div>
            <div class="subtle">{T('create_pf')}</div></div>""", unsafe_allow_html=True)
    _, c2, _ = st.columns([1, 1, 1])
    with c2:
        if st.button(T("create_btn"), use_container_width=True):
            new_portfolio(); st.rerun()


# ======================================================================================
# CONFIGURE
# ======================================================================================
def render_feature(pid, feat):
    spec = SPEC[feat]
    st.markdown(_label(feat), unsafe_allow_html=True)
    slider_opts = ["N/A"] + list(spec["options"])

    if spec["type"] == "nominal":
        choice = st.selectbox(" ", slider_opts, key=f"in_{pid}_{feat}", label_visibility="collapsed")
        return None if choice == "N/A" else spec["value_map"][choice]

    if spec["type"] == "numeric":
        c1, c2 = st.columns([0.76, 0.24])
        with c1:
            choice = st.select_slider(" ", options=slider_opts, value="N/A",
                                      key=f"in_{pid}_{feat}", label_visibility="collapsed")
        with c2:
            raw = st.text_input(" ", key=f"num_{pid}_{feat}", placeholder=T("custom"),
                                label_visibility="collapsed")
        st.markdown(_ticks(slider_opts), unsafe_allow_html=True)
        if raw:
            try:
                return float(raw.replace(",", "."))
            except ValueError:
                pass
        return None if choice == "N/A" else spec["value_map"][choice]

    choice = st.select_slider(" ", options=slider_opts, value="N/A",
                              key=f"in_{pid}_{feat}", label_visibility="collapsed")
    st.markdown(_ticks(slider_opts), unsafe_allow_html=True)
    return None if choice == "N/A" else spec["value_map"][choice]


def render_category_aggregate(pid, crit, feats):
    st.markdown(f"<div class='param-label'><span title='{T('tip_agg')}'>{T('overall')} {crit} &#9432;</span>"
                f"{_mini(_cat_dir(feats))}</div>", unsafe_allow_html=True)
    choice = st.select_slider(" ", options=AGG_LEVELS, value="N/A",
                              key=f"agg_{pid}_{crit}", label_visibility="collapsed")
    st.markdown(_ticks(AGG_LEVELS), unsafe_allow_html=True)
    if choice == "N/A":
        return {f: None for f in feats}
    frac = (AGG_LEVELS.index(choice) - 1) / (len(AGG_LEVELS) - 2)
    return {f: SPEC[f]["value_map"][SPEC[f]["options"][round(frac * (len(SPEC[f]["options"]) - 1))]]
            for f in feats}


def render_configure(pf):
    st.markdown(f"## {T('pf_config')}")
    pf["name"] = st.text_input(T("pf_name"), value=pf["name"])

    if pf["projects"]:
        st.markdown(f"<span class='subtle'>{T('added_projects')}</span>", unsafe_allow_html=True)
        for i, proj in enumerate(pf["projects"]):
            row_l, row_r = st.columns([0.86, 0.14])
            row_l.markdown(f"<div style='padding:0.35rem 0; color:{TEXT};'>{proj['name']}</div>",
                           unsafe_allow_html=True)
            if row_r.button(T("remove"), key=f"del_{i}"):
                pf["projects"].pop(i); st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pid = ss.draft_id
    with st.container(border=True):
        proj_name = st.text_input(T("project_name"), value=f"Project {len(pf['projects']) + 1}",
                                  key=f"pname_{pid}")
        st.caption(T("min_hint", n=MIN_FEATURES))

        draft = {}
        for crit, feats in mb.KUB_GROUPS.items():
            with st.container(border=True):
                h1, h2 = st.columns([0.6, 0.4])
                h1.markdown(f"<div class='cat-header' style='padding-top:0.4rem;'>{crit}</div>",
                            unsafe_allow_html=True)
                with h2:
                    detailed = st.toggle(T("single"), key=f"tg_{pid}_{crit}")
                if detailed:
                    for f in feats:
                        draft[f] = render_feature(pid, f)
                else:
                    draft.update(render_category_aggregate(pid, crit, feats))

    n_set = sum(v is not None for v in draft.values())
    _, a_col, b_col, _ = st.columns([0.28, 0.22, 0.22, 0.28])
    with a_col:
        if st.button(f"{T('add_project')} ({n_set})", use_container_width=True):
            if n_set < MIN_FEATURES:
                st.warning(T("warn_min", n=MIN_FEATURES, c=n_set))
            else:
                pf["projects"].append({"name": proj_name, "params": draft})
                ss.draft_id += 1; st.rerun()
    with b_col:
        if st.button(T("calc"), use_container_width=True):
            if not pf["projects"]:
                st.warning(T("warn_add"))
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
                 f"<div style='width:190px; font-size:0.82rem; color:{TEXT};'>{mb.label_of(feat)}{tag}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:5px; height:10px;'>"
                 f"<div style='width:{abs(v)/maxabs*100:.0f}%; background:{c}; height:10px; border-radius:5px;'></div></div>"
                 f"<div style='width:56px; text-align:right; color:{c}; font-size:0.8rem; font-weight:600;'>{v:+.2f}</div>"
                 f"</div>")
    return (f"<div class='subtle' style='font-size:0.75rem; margin-bottom:0.3rem;'>{T('shap_note')}</div>") + rows


def render_project_card(proj):
    order, proba = mb.predict(MODEL, META, proj["params"])
    pred = order[list(proba).index(max(proba))]
    color = LEVEL_COLORS[pred]
    score = mb.expected_score(order, proba)
    n_set = sum(v is not None for v in proj["params"].values())

    with st.container(border=True):
        st.markdown(
            f"""<div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div style="font-size:1.1rem; font-weight:600; color:{TEXT};">{proj['name']}</div>
                <div style="color:{color}; font-weight:700;">{pred}</div></div>
            <div class="info" style="margin:0.1rem 0 0.7rem 0; color:{MUTED}; font-size:0.9rem;">
                <span title="{T('tip_score')}">{T('exp_score')}: {score:.2f} &#9432;</span>
                &middot; {n_set} {T('features_set')}</div>
            {prob_bars(order, proba, pred)}""", unsafe_allow_html=True)
        with st.expander(T("expander")):
            st.markdown(f"<div class='cat-header'>{T('top_drivers')}</div>", unsafe_allow_html=True)
            st.markdown(shap_drivers(order, proba, proj["params"]), unsafe_allow_html=True)
            st.markdown(f"<div class='cat-header' style='margin-top:1rem;'>{T('set_params')}</div>",
                        unsafe_allow_html=True)
            rows = ""
            for feat, v in proj["params"].items():
                if v is not None:
                    val = round(v, 2) if isinstance(v, float) else v
                    rows += (f"<div style='display:flex; justify-content:space-between; padding:0.2rem 0;"
                             f" border-bottom:1px solid {BORDER};'>"
                             f"<span style='color:{MUTED};'>{mb.label_of(feat)}</span>"
                             f"<span style='font-weight:600; color:{TEXT};'>{val}</span></div>")
            st.markdown(rows or f"<span class='subtle'>{T('no_set')}</span>", unsafe_allow_html=True)


def render_results(pf):
    st.markdown(f"## {T('risk_results')}")
    st.markdown(f"<div class='subtle'>{pf['name']}</div>", unsafe_allow_html=True)
    if not pf["projects"]:
        st.info(T("no_projects"))
        return

    per_project = [mb.predict(MODEL, META, p["params"]) for p in pf["projects"]]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]

    with st.container(border=True):
        st.markdown(
            f"""<div style="font-size:1.1rem; font-weight:600; margin-bottom:1rem; color:{TEXT};">{T('pf_summary')}</div>
            <div class="info" style="display:flex; gap:2.5rem; flex-wrap:wrap;">
                <div><div class="subtle">{T('total_risk')}</div>
                    <div style="font-size:1.9rem; font-weight:700; color:{p_color};">{pm['level']}</div></div>
                <div style="margin-left:auto; text-align:right;">
                    <div class="subtle"><span title="{T('tip_elev')}">{T('p_elev')} &#9432;</span></div>
                    <div style="font-size:1.5rem; font-weight:600; color:{TEXT};">{pm['p_at_least_one_elevated']:.0%}</div></div>
                <div style="text-align:right;">
                    <div class="subtle"><span title="{T('tip_cnt')}">{T('exp_high')} &#9432;</span></div>
                    <div style="font-size:1.5rem; font-weight:600; color:{TEXT};">{pm['expected_elevated_count']:.1f}</div></div>
                <div style="text-align:right;"><div class="subtle">{T('projects')}</div>
                    <div style="font-size:1.5rem; font-weight:600; color:{TEXT};">{pm['n']}</div></div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"### {T('breakdown')}")
    for proj in pf["projects"]:
        render_project_card(proj)


# ======================================================================================
# ROUTER
# ======================================================================================
nav_col, _ = st.columns([0.4, 0.6])
with nav_col:
    view_labels = [T("configure"), T("results")]
    sel = st.radio("nav", view_labels, index=0 if ss.view == "Configure" else 1,
                   horizontal=True, label_visibility="collapsed")
    ss.view = "Configure" if sel == view_labels[0] else "Results"
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if ss.active is None or ss.active not in ss.portfolios:
    render_empty_state()
elif ss.view == "Configure":
    render_configure(ss.portfolios[ss.active])
else:
    render_results(ss.portfolios[ss.active])
