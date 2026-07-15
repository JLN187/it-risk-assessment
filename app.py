"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import streamlit as st
import model_bridge as mb

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------
BG, PANEL, PANEL_2, BORDER = "#262624", "#31312e", "#3c3c38", "#55554f"
TEXT, MUTED, HEAD = "#f2f0ea", "#b3b0a8", "#d8d5cd"
GREEN, RED = "#5bb56b", "#e0574f"
LEVEL_COLORS = {"Low": GREEN, "Medium": "#d9a441", "High": "#e5844d", "Critical": RED, "N/A": MUTED}
AGG_EN = ["N/A", "Low", "Medium", "High", "Very High"]

# --------------------------------------------------------------------------------------
# i18n
# --------------------------------------------------------------------------------------
STR = {
 "en": {"app_title": "Portfolio Risk Analyzer", "configure": "Configure", "results": "Results",
   "portfolios": "Portfolios", "new_portfolio": "New Portfolio", "no_portfolios": "No portfolios yet.",
   "no_pf_sel": "No Portfolio Selected", "create_pf": "Create a new portfolio to get started",
   "create_btn": "Create New Portfolio", "pf_config": "Portfolio Configuration", "pf_name": "Portfolio Name",
   "added_projects": "Added Projects", "project_name": "Project Name",
   "min_hint": "Set at least {n} features. The more features you set, the more reliable the prediction.",
   "single": "Details", "fine_tune": "Set individually", "overall": "Overall", "add_project": "Add Project", "calc": "Calculate Results",
   "warn_min": "Please set at least {n} features (currently {c}).", "warn_add": "Please add at least one project first.",
   "risk_results": "Risk Assessment Results", "pf_summary": "Portfolio Risk Summary", "total_risk": "Total Portfolio Risk",
   "p_elev": "P(\u22651 elevated-risk project)", "exp_high": "Expected # \u2265 High", "projects": "Projects",
   "breakdown": "Project Risk Breakdown", "no_projects": "No projects yet. Add one in Configure and calculate.",
   "drivers_up": "INCREASES RISK", "drivers_down": "DECREASES RISK",
   "shap_note": "SHAP contribution to overall risk (risk-weighted across classes).",
   "default_expl": "Features you didn't set use the dataset's typical value \u2014 which still influences the result. Such drivers are marked (default).",
   "set_params": "Set parameters", "exp_score": "Expected score", "features_set": "feature(s) set",
   "custom": "custom", "levels": "Levels (left \u2192 right): ", "reliability": "Reliability",
   "rel_low": "Low", "rel_med": "Medium", "rel_high": "High",
   "rel_tip": "Heuristic based on how many features you set (not a statistical confidence interval). More inputs = the prediction rests less on dataset defaults.",
   "dir_more": "more risk", "dir_less": "less risk",
   "tip_score": "Expected value of the risk level (Low=1..Critical=4), weighted by class probabilities.",
   "tip_elev": "Probability that at least one project is High or Critical risk. Assumes projects are independent (probability-tree path).",
   "tip_cnt": "Expected number of projects at High or Critical risk (sum of individual probabilities).",
   "tip_agg": "How much risk this category contributes overall. Each feature in it is set to a matching value (features where a higher value means less risk are mapped inversely).",
   "load_err": "Model artifacts could not be loaded. Please run masterskript_final.py locally first."},
 "de": {"app_title": "Portfolio-Risikoanalyse", "configure": "Konfigurieren", "results": "Ergebnisse",
   "portfolios": "Portfolios", "new_portfolio": "Neues Portfolio", "no_portfolios": "Noch keine Portfolios.",
   "no_pf_sel": "Kein Portfolio ausgew\u00e4hlt", "create_pf": "Erstelle ein Portfolio, um zu starten",
   "create_btn": "Neues Portfolio erstellen", "pf_config": "Portfolio-Konfiguration", "pf_name": "Portfolioname",
   "added_projects": "Hinzugef\u00fcgte Projekte", "project_name": "Projektname",
   "min_hint": "Mindestens {n} Merkmale angeben. Je mehr Merkmale gesetzt sind, desto zuverl\u00e4ssiger die Vorhersage.",
   "single": "Details", "fine_tune": "Einzeln einstellen", "overall": "Gesamt", "add_project": "Projekt hinzuf\u00fcgen", "calc": "Ergebnisse berechnen",
   "warn_min": "Bitte mindestens {n} Merkmale setzen (aktuell {c}).", "warn_add": "Bitte zuerst mindestens ein Projekt hinzuf\u00fcgen.",
   "risk_results": "Risikobewertung", "pf_summary": "Portfolio-Risiko\u00fcbersicht", "total_risk": "Gesamt-Portfoliorisiko",
   "p_elev": "P(\u22651 Hochrisikoprojekt)", "exp_high": "Erwartete Anzahl \u2265 High", "projects": "Projekte",
   "breakdown": "Projekt-Risiko im Detail", "no_projects": "Noch keine Projekte. F\u00fcge unter Konfigurieren eins hinzu und berechne.",
   "drivers_up": "ERH\u00d6HT RISIKO", "drivers_down": "SENKT RISIKO",
   "shap_note": "SHAP-Beitrag zum Gesamtrisiko (klassen\u00fcbergreifend risikogewichtet).",
   "default_expl": "Nicht gesetzte Merkmale verwenden den datensatztypischen Wert \u2014 der das Ergebnis trotzdem beeinflusst. Solche Treiber sind mit (Default) markiert.",
   "set_params": "Gesetzte Merkmale", "exp_score": "Erwartungswert-Score", "features_set": "Merkmal(e) gesetzt",
   "custom": "eigener Wert", "levels": "Stufen (links \u2192 rechts): ", "reliability": "Zuverl\u00e4ssigkeit",
   "rel_low": "Gering", "rel_med": "Mittel", "rel_high": "Hoch",
   "rel_tip": "Heuristik basierend auf der Anzahl gesetzter Merkmale (kein statistisches Konfidenzintervall). Mehr Eingaben = die Vorhersage st\u00fctzt sich weniger auf datensatztypische Standardwerte.",
   "dir_more": "mehr Risiko", "dir_less": "weniger Risiko",
   "tip_score": "Erwartungswert der Risikostufe (Low=1..Critical=4), gewichtet mit den Klassenwahrscheinlichkeiten.",
   "tip_elev": "Wahrscheinlichkeit, dass mindestens ein Projekt High- oder Critical-Risiko hat. Annahme: Projekte unabh\u00e4ngig (Baumdiagramm-Pfad).",
   "tip_cnt": "Erwartete Anzahl Projekte mit High- oder Critical-Risiko (Summe der Einzelwahrscheinlichkeiten).",
   "tip_agg": "Wie viel Risiko diese Kategorie insgesamt beitr\u00e4gt. Jedes Merkmal darin wird auf einen passenden Wert gesetzt (Merkmale, bei denen ein h\u00f6herer Wert weniger Risiko bedeutet, werden gespiegelt abgebildet).",
   "load_err": "Modell-Artefakte konnten nicht geladen werden. Bitte zuerst masterskript_final.py lokal ausf\u00fchren."},
}
DE_CAT = {"Complexity": "Komplexit\u00e4t", "Efficiency": "Effizienz", "Risk": "Risiko",
          "Strategy": "Strategie", "Urgency": "Dringlichkeit"}
DE_LABEL = {
 "Complexity_Score": "Komplexit\u00e4tswert", "Integration_Complexity": "Integrationskomplexit\u00e4t",
 "Cross_Functional_Dependencies": "Bereichs\u00fcbergreifende Abh\u00e4ngigkeiten",
 "External_Dependencies_Count": "Externe Abh\u00e4ngigkeiten (Anzahl)", "Technology_Familiarity": "Technologie-Vertrautheit",
 "Tech_Environment_Stability": "Stabilit\u00e4t der Technikumgebung", "Technical_Debt_Level": "Technische Schulden",
 "Requirement_Stability": "Anforderungsstabilit\u00e4t", "Change_Request_Frequency": "H\u00e4ufigkeit \u00c4nderungsanfragen",
 "Team_Size": "Teamgr\u00f6\u00dfe", "Stakeholder_Count": "Anzahl Stakeholder", "Geographical_Distribution": "Geografische Verteilung",
 "Project_Budget_USD": "Projektbudget", "Budget_Utilization_Rate": "Budgetauslastung",
 "Estimated_Timeline_Months": "Gesch\u00e4tzte Laufzeit (Monate)", "Resource_Availability": "Ressourcenverf\u00fcgbarkeit",
 "Resource_Contention_Level": "Ressourcenkonkurrenz", "Current_Phase_Duration_Months": "Dauer aktuelle Phase (Monate)",
 "Communication_Frequency": "Kommunikationsfrequenz", "Documentation_Quality": "Dokumentationsqualit\u00e4t",
 "Org_Process_Maturity": "Prozessreife der Organisation", "Team_Experience_Level": "Team-Erfahrungsniveau",
 "Project_Manager_Experience": "Projektleiter-Erfahrung", "Past_Similar_Projects": "Fr\u00fchere \u00e4hnliche Projekte",
 "Previous_Delivery_Success_Rate": "Bisherige Liefererfolgsquote", "Methodology_Used": "Vorgehensmodell",
 "Team_Colocation": "Team-Verteilung", "Historical_Risk_Incidents": "Fr\u00fchere Risikovorf\u00e4lle",
 "Risk_Management_Maturity": "Reife des Risikomanagements", "Change_Control_Maturity": "Reife der \u00c4nderungssteuerung",
 "Vendor_Reliability_Score": "Zuverl\u00e4ssigkeit der Dienstleister", "Team_Turnover_Rate": "Personalfluktuation",
 "Market_Volatility": "Marktvolatilit\u00e4t", "Industry_Volatility": "Branchenvolatilit\u00e4t",
 "Regulatory_Compliance_Level": "Regulatorische Anforderungen", "Data_Security_Requirements": "Datensicherheitsanforderungen",
 "Seasonal_Risk_Factor": "Saisonaler Risikofaktor", "Executive_Sponsorship": "Management-Unterst\u00fctzung",
 "Stakeholder_Engagement_Level": "Stakeholder-Einbindung", "Key_Stakeholder_Availability": "Verf\u00fcgbarkeit zentraler Stakeholder",
 "Funding_Source": "Finanzierungsquelle", "Contract_Type": "Vertragsart", "Client_Experience_Level": "Kundenerfahrung",
 "Organizational_Change_Frequency": "H\u00e4ufigkeit organisatorischer \u00c4nderungen", "Priority_Level": "Priorit\u00e4t",
 "Schedule_Pressure": "Zeitdruck", "Project_Phase": "Projektphase", "Project_Start_Month": "Projektstartmonat (1-12)"}
VALUE_DE = {
 "Low": "Niedrig", "Medium": "Mittel", "High": "Hoch", "Very High": "Sehr hoch", "Critical": "Kritisch",
 "Volatile": "Volatil", "Moderate": "Moderat", "Stable": "Stabil", "New": "Neu", "Familiar": "Vertraut",
 "Expert": "Experte", "Poor": "Schlecht", "Excellent": "Ausgezeichnet", "Weak": "Schwach", "Strong": "Stark",
 "Ad-hoc": "Ad-hoc", "Defined": "Definiert", "Managed": "Gesteuert", "Optimizing": "Optimierend", "Strict": "Streng",
 "Limited": "Begrenzt", "Good": "Gut", "Legacy/Unstable": "Alt/Instabil", "Mixed": "Gemischt",
 "Modern/Stable": "Modern/Stabil", "Extreme": "Extrem", "Basic": "Basis", "Formal": "Formal", "Advanced": "Fortgeschritten",
 "Junior": "Junior", "Senior": "Senior", "Junior PM": "Junior-PM", "Mid-level PM": "Mittleres PM",
 "Senior PM": "Senior-PM", "Certified PM": "Zertifiziertes PM", "First-time": "Erstmalig", "Occasional": "Gelegentlich",
 "Regular": "Regelm\u00e4\u00dfig", "Strategic": "Strategisch", "Agile": "Agil", "Kanban": "Kanban", "Scrum": "Scrum",
 "External": "Extern", "Government": "Staatlich", "Internal": "Intern", "Cost-Plus": "Cost-Plus",
 "Fixed-Price": "Festpreis", "Hybrid": "Hybrid", "Time & Materials": "Zeit & Material", "Closure": "Abschluss",
 "Execution": "Ausf\u00fchrung", "Initiation": "Initiierung", "Monitoring": "\u00dcberwachung", "Planning": "Planung",
 "Fully Colocated": "Voll vor Ort", "Fully Remote": "Voll remote", "Partially Colocated": "Teilweise vor Ort"}


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

st.set_page_config(page_title="Portfolio Risk Assessment", layout="wide", initial_sidebar_state="expanded")

# Minimal-CSS: Theme uebernimmt Widgets; hier nur Rahmen/Feinheiten + lesbarer Aktiv-Button
st.markdown(f"""
<style>
 .block-container {{ padding-top:4rem; padding-bottom:4rem; max-width:1320px; }}
 h1,h2,h3 {{ font-weight:600; letter-spacing:-0.01em; }}
 .subtle {{ color:{MUTED}; font-size:0.95rem; }}
 .cat-header {{ color:{HEAD}; text-transform:uppercase; letter-spacing:0.12em; font-size:0.8rem; font-weight:700; }}
 .param-label {{ font-weight:600; font-size:0.88rem; margin:0.5rem 0 0.1rem 0; color:{TEXT}; }}
 .param-label span, .info span {{ cursor:help; }}
 .tick-row {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:0.68rem; margin:0 0 0.55rem 0; }}
 .tick-marks {{ display:flex; justify-content:space-between; margin:-0.55rem 6px 0.1rem 6px; }}
 .tick-marks span {{ width:1px; height:6px; background:{BORDER}; display:block; }}
 .divider {{ height:1px; background:{BORDER}; margin:1rem 0; border:none; }}
 /* klar sichtbare Karten-Rahmen + verschachtelte Tiefe */
 [data-testid="stVerticalBlockBorderWrapper"] {{ border:1px solid {BORDER} !important; border-radius:10px !important; background:{PANEL}; }}
 [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {{ background:{BG}; }}
 /* aktiver Nav-/Sprach-Button: dunkles Highlight mit HELLEM Text (klarer Kontrast) */
 .stButton > button[kind="primary"] {{ background:{PANEL_2} !important; color:{TEXT} !important; border:1px solid {HEAD} !important; }}
 .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] div {{ color:{TEXT} !important; }}
 div[data-baseweb="slider"] div[role="slider"] {{ background:{HEAD} !important; border:2px solid {TEXT} !important; box-shadow:0 0 0 4px rgba(216,213,205,0.20) !important; }}
 [data-testid="stThumbValue"] {{ color:{TEXT} !important; font-weight:700 !important; }}
</style>
""", unsafe_allow_html=True)

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


def L(feat):
    return DE_LABEL.get(feat, feat.replace("_", " ")) if ss.lang == "de" else mb.label_of(feat)


def CAT(crit):
    return DE_CAT.get(crit, crit) if ss.lang == "de" else crit


def vopt(v):
    return VALUE_DE.get(v, v) if ss.lang == "de" else v


def new_portfolio():
    ss.pf_counter += 1
    pid = f"pf{ss.pf_counter}"
    ss.portfolios[pid] = {"name": f"Portfolio {ss.pf_counter}", "projects": []}
    ss.active, ss.view = pid, "Configure"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<div style='font-size:1.3rem; font-weight:700; color:{TEXT}; margin-bottom:0.6rem;'>"
                f"{T('app_title')}</div>", unsafe_allow_html=True)
    lc1, lc2, _ = st.columns([0.28, 0.28, 0.44])
    if lc1.button("\U0001F1EC\U0001F1E7", key="lang_en", use_container_width=True,
                  type="primary" if ss.lang == "en" else "secondary"):
        ss.lang = "en"; st.rerun()
    if lc2.button("\U0001F1E9\U0001F1EA", key="lang_de", use_container_width=True,
                  type="primary" if ss.lang == "de" else "secondary"):
        ss.lang = "de"; st.rerun()
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"**{T('portfolios')}**")
    if st.button("\uFF0B  " + T("new_portfolio"), use_container_width=True):
        new_portfolio(); st.rerun()
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
    marks = "<div class='tick-marks'>" + "".join("<span></span>" for _ in opts) + "</div>"
    labels = "<div class='tick-row'>" + "".join(f"<span>{o}</span>" for o in opts) + "</div>"
    return marks + labels


def _grad(d):
    return (f"to right,{GREEN},{RED}" if d > 0 else f"to right,{RED},{GREEN}")


def _mini(feat):
    d = SPEC[feat]["direction"]
    if d == 0:
        return ""
    opts = SPEC[feat]["options"]
    lo, hi = vopt(opts[0]), vopt(opts[-1])
    tip = (f"{lo} \u2192 {T('dir_less')}, {hi} \u2192 {T('dir_more')}" if d > 0
           else f"{lo} \u2192 {T('dir_more')}, {hi} \u2192 {T('dir_less')}")
    return (f"<span title='{tip}' style='display:inline-block; width:26px; height:6px; border-radius:3px; "
            f"vertical-align:middle; margin-left:8px; background:linear-gradient({_grad(d)});'></span>")


def _mini_cat(feats):
    # Aggregat ist immer in Risiko-Semantik: links = wenig Risiko, rechts = viel Risiko
    return (f"<span title='{T('tip_agg')}' style='display:inline-block; width:26px; height:6px; "
            f"border-radius:3px; vertical-align:middle; margin-left:8px; "
            f"background:linear-gradient(to right,{GREEN},{RED});'></span>")


def _tip(feat):
    return T("levels") + ", ".join(vopt(o) for o in SPEC[feat]["options"])


def reliability(n):
    if n < 15:  return T("rel_low"), "#e5844d"
    if n < 30:  return T("rel_med"), "#d9a441"
    return T("rel_high"), GREEN


# ======================================================================================
# EMPTY STATE
# ======================================================================================
def render_empty_state():
    st.markdown(f"""<div style="text-align:center; padding:6rem 0;">
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
    st.markdown(f"<div class='param-label'><span title='{_tip(feat)}'>{L(feat)} &#9432;</span>{_mini(feat)}</div>",
                unsafe_allow_html=True)
    disp = ["N/A"] + [vopt(o) for o in spec["options"]]
    back = {vopt(o): o for o in spec["options"]}
    if spec["type"] == "nominal":
        choice = st.selectbox(" ", disp, key=f"in_{pid}_{feat}", label_visibility="collapsed")
        return None if choice == "N/A" else spec["value_map"][back[choice]]
    if spec["type"] == "numeric":
        c1, c2 = st.columns([0.76, 0.24])
        with c1:
            choice = st.select_slider(" ", options=disp, value="N/A", key=f"in_{pid}_{feat}", label_visibility="collapsed")
        with c1:
            st.markdown(_ticks(disp), unsafe_allow_html=True)
        with c2:
            raw = st.text_input(" ", key=f"num_{pid}_{feat}", placeholder=T("custom"), label_visibility="collapsed",
                                help=f"{spec['min']:g} \u2013 {spec['max']:g}")
        if raw:
            try:
                val = float(raw.replace(",", "."))
                return max(spec["min"], min(spec["max"], val))   # auf Slider-Grenzen begrenzen
            except ValueError:
                pass
        return None if choice == "N/A" else spec["value_map"][back[choice]]
    choice = st.select_slider(" ", options=disp, value="N/A", key=f"in_{pid}_{feat}", label_visibility="collapsed")
    st.markdown(_ticks(disp), unsafe_allow_html=True)
    return None if choice == "N/A" else spec["value_map"][back[choice]]


def render_category_aggregate(pid, crit, feats):
    st.markdown(f"<div class='param-label'><span title='{T('tip_agg')}'>{T('overall')} {CAT(crit)} &#9432;</span>"
                f"{_mini_cat(feats)}</div>", unsafe_allow_html=True)
    disp = ["N/A"] + [vopt(x) for x in AGG_EN[1:]]
    choice = st.select_slider(" ", options=disp, value="N/A", key=f"agg_{pid}_{crit}", label_visibility="collapsed")
    st.markdown(_ticks(disp), unsafe_allow_html=True)
    idx = disp.index(choice)
    if idx == 0:
        return {f: None for f in feats}
    frac = (idx - 1) / (len(disp) - 2)          # 0 = wenig Risiko ... 1 = viel Risiko
    out = {}
    for f in feats:
        o = SPEC[f]["options"]
        # Merkmale, bei denen "hoeher = weniger Risiko" gilt, gespiegelt abbilden
        ff = frac if SPEC[f]["direction"] >= 0 else (1 - frac)
        out[f] = SPEC[f]["value_map"][o[round(ff * (len(o) - 1))]]
    return out


def render_configure(pf):
    st.markdown(f"## {T('pf_config')}")
    pf["name"] = st.text_input(T("pf_name"), value=pf["name"])

    if pf["projects"]:
        st.markdown(f"<span class='subtle'>{T('added_projects')}</span>", unsafe_allow_html=True)
        for i, proj in enumerate(pf["projects"]):
            row_l, row_r = st.columns([0.88, 0.12])
            row_l.markdown(f"<div style='padding:0.45rem 0.2rem; color:{TEXT};'>{proj['name']}</div>",
                           unsafe_allow_html=True)
            if row_r.button("\U0001F5D1", key=f"del_{i}"):
                pf["projects"].pop(i); st.rerun()
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    pid = ss.draft_id
    with st.container(border=True):
        proj_name = st.text_input(T("project_name"), value=f"Project {len(pf['projects']) + 1}", key=f"pname_{pid}")
        st.caption(T("min_hint", n=MIN_FEATURES))
        draft = {}
        for crit, feats in mb.KUB_GROUPS.items():
            with st.container(border=True):
                h1, h2 = st.columns([0.7, 0.3])
                h1.markdown(f"<div class='cat-header' style='padding-top:0.45rem;'>{CAT(crit)}</div>",
                            unsafe_allow_html=True)
                with h2:
                    detailed = st.toggle(T("fine_tune"), key=f"tg_{pid}_{crit}")
                if detailed:
                    for f in feats:
                        draft[f] = render_feature(pid, f)
                else:
                    draft.update(render_category_aggregate(pid, crit, feats))

    n_set = sum(v is not None for v in draft.values())
    _, a_col, b_col, _ = st.columns([0.26, 0.24, 0.24, 0.26])
    with a_col:
        if st.button(f"{T('add_project')} ({n_set})", use_container_width=True, type="primary"):
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
                 f"<div style='width:70px; color:{MUTED}; font-size:0.85rem;'>{vopt(cls)}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:6px; height:14px;'>"
                 f"<div style='width:{p*100:.1f}%; background:{c}; height:14px; border-radius:6px;'></div></div>"
                 f"<div style='width:52px; text-align:right; color:{c}; {strong} font-size:0.85rem;'>{p:.1%}</div></div>")
    return rows


def _driver_rows(items, maxabs):
    rows = ""
    for feat, v, is_set in items:
        c = RED if v > 0 else GREEN
        tag = "" if is_set else f" <span style='color:{MUTED}; font-size:0.72rem;'>(default)</span>"
        rows += (f"<div style='display:flex; align-items:center; gap:0.7rem; margin:0.2rem 0;'>"
                 f"<div style='width:190px; font-size:0.8rem; color:{TEXT};'>{L(feat)}{tag}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:5px; height:9px;'>"
                 f"<div style='width:{abs(v)/maxabs*100:.0f}%; background:{c}; height:9px; border-radius:5px;'></div></div>"
                 f"<div style='width:52px; text-align:right; color:{c}; font-size:0.78rem; font-weight:600;'>{v:+.2f}</div></div>")
    return rows


def render_project_details(order, proba, params):
    drivers = mb.explain(CTX, META, params, order, proba)
    pos = [d for d in drivers if d[1] > 0][:5]
    neg = sorted([d for d in drivers if d[1] < 0], key=lambda x: x[1])[:5]
    maxabs = max([abs(v) for _, v, _ in pos + neg], default=1) or 1
    st.markdown(f"<div class='subtle' style='font-size:0.75rem;'>{T('shap_note')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='cat-header' style='color:{RED};'>{T('drivers_up')}</div>", unsafe_allow_html=True)
    st.markdown(_driver_rows(pos, maxabs) or "<span class='subtle'>\u2014</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='cat-header' style='color:{GREEN}; margin-top:0.6rem;'>{T('drivers_down')}</div>",
                unsafe_allow_html=True)
    st.markdown(_driver_rows(neg, maxabs) or "<span class='subtle'>\u2014</span>", unsafe_allow_html=True)
    st.caption(T("default_expl"))
    st.markdown(f"<div class='cat-header' style='margin-top:0.8rem;'>{T('set_params')}</div>", unsafe_allow_html=True)
    for crit, feats in mb.KUB_GROUPS.items():
        setf = [(f, params[f]) for f in feats if params.get(f) is not None]
        if not setf:
            continue
        with st.expander(CAT(crit)):
            rows = ""
            for f, v in setf:
                val = round(v, 2) if isinstance(v, float) else vopt(v)
                rows += (f"<div style='display:flex; justify-content:space-between; padding:0.2rem 0;"
                         f" border-bottom:1px solid {BORDER};'>"
                         f"<span style='color:{MUTED};'>{L(f)}</span>"
                         f"<span style='font-weight:600; color:{TEXT};'>{val}</span></div>")
            st.markdown(rows, unsafe_allow_html=True)


def render_project_card(i, proj):
    order, proba = mb.predict(MODEL, META, proj["params"])
    pred = order[list(proba).index(max(proba))]
    color = LEVEL_COLORS[pred]
    score = mb.expected_score(order, proba)
    n_set = sum(v is not None for v in proj["params"].values())
    rel_label, rel_col = reliability(n_set)
    with st.container(border=True):
        head_l, head_r = st.columns([0.75, 0.25])
        head_l.markdown(f"<div style='font-size:1.1rem; font-weight:600; color:{TEXT};'>{proj['name']}</div>"
                        f"<div class='info' style='color:{MUTED}; font-size:0.85rem; margin-top:0.1rem;'>"
                        f"<span title='{T('tip_score')}'>{T('exp_score')}: {score:.2f} &#9432;</span> &middot; "
                        f"<span title='{T('rel_tip')}'>{T('reliability')}: "
                        f"<span style='color:{rel_col}; font-weight:600;'>{rel_label}</span> &#9432;</span></div>",
                        unsafe_allow_html=True)
        head_r.markdown(f"<div style='text-align:right; color:{color}; font-weight:700; font-size:1.05rem;'>{pred}</div>",
                        unsafe_allow_html=True)
        show = head_r.toggle(T("single"), key=f"pdet_{ss.active}_{i}")
        st.markdown(prob_bars(order, proba, pred) + "<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        if show:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            render_project_details(order, proba, proj["params"])


def render_results(pf):
    st.markdown(f"## {T('risk_results')}")
    st.markdown(f"<div class='subtle'>{pf['name']}</div>", unsafe_allow_html=True)
    if not pf["projects"]:
        st.info(T("no_projects")); return
    per_project = [mb.predict(MODEL, META, p["params"]) for p in pf["projects"]]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]
    with st.container(border=True):
        st.markdown(
            f"""<div style="font-size:1.1rem; font-weight:600; margin-bottom:1rem; color:{TEXT};">{T('pf_summary')}</div>
            <div class="info" style="display:flex; gap:2.5rem; flex-wrap:wrap;">
                <div><div class="subtle">{T('total_risk')}</div>
                    <div style="font-size:1.9rem; font-weight:700; color:{p_color};">{vopt(pm['level'])}</div></div>
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
    for i, proj in enumerate(pf["projects"]):
        render_project_card(i, proj)


# ======================================================================================
# ROUTER (mittige Segment-Navigation aus zwei abgerundeten Buttons)
# ======================================================================================
_, mid, _ = st.columns([0.26, 0.48, 0.26])
with mid:
    n1, n2 = st.columns(2)
    if n1.button(T("configure"), use_container_width=True,
                 type="primary" if ss.view == "Configure" else "secondary"):
        ss.view = "Configure"; st.rerun()
    if n2.button(T("results"), use_container_width=True,
                 type="primary" if ss.view == "Results" else "secondary"):
        ss.view = "Results"; st.rerun()
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

if ss.active is None or ss.active not in ss.portfolios:
    render_empty_state()
elif ss.view == "Configure":
    render_configure(ss.portfolios[ss.active])
else:
    render_results(ss.portfolios[ss.active])
