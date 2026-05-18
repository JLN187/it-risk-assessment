"""
IT Portfolio Risk Assessment
Prototyp zur Vorhersage von IT-Projektrisikoniveaus mittels ML
Aufbauend auf: Karrenbauer & Breitner (2022)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="IT Portfolio Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Hintergrund & Basis */
    .stApp { background-color: #212121; color: #ececec; }
    section[data-testid="stSidebar"] { background-color: #171717; }

    /* Metrikkarten */
    .metric-box {
        background: #2f2f2f;
        border: 1px solid #3d3d3d;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
    }
    .metric-title { font-size: 12px; color: #8e8ea0; margin-bottom: 6px; letter-spacing: 0.04em; }
    .metric-value { font-size: 24px; font-weight: 700; color: #ececec; }

    /* Risikoklassen-Badges */
    .risk-low      { background:#1a3a2a; color:#4ade80; border:1px solid #166534;
                     padding:5px 14px; border-radius:20px; font-weight:600;
                     display:inline-block; font-size:14px; }
    .risk-medium   { background:#3a2e0a; color:#facc15; border:1px solid #713f12;
                     padding:5px 14px; border-radius:20px; font-weight:600;
                     display:inline-block; font-size:14px; }
    .risk-high     { background:#3a1a1a; color:#f87171; border:1px solid #7f1d1d;
                     padding:5px 14px; border-radius:20px; font-weight:600;
                     display:inline-block; font-size:14px; }
    .risk-critical { background:#4a0d0d; color:#fca5a5; border:1px solid #991b1b;
                     padding:5px 14px; border-radius:20px; font-weight:600;
                     display:inline-block; font-size:14px; }

    /* Eingabebereich */
    .input-card {
        background: #2f2f2f;
        border: 1px solid #3d3d3d;
        border-radius: 16px;
        padding: 28px 32px;
        margin: 0 auto;
    }

    /* Projektzeile */
    .projekt-row {
        background: #2f2f2f;
        border: 1px solid #3d3d3d;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    /* Streamlit Elemente anpassen */
    .stSlider > div > div { background: #3d3d3d; }
    .stTextInput input {
        background: #2f2f2f !important;
        border: 1px solid #4d4d4d !important;
        color: #ececec !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton button:hover { background: #0d8a6c !important; }
    div[data-testid="stExpander"] {
        background: #2f2f2f;
        border: 1px solid #3d3d3d;
        border-radius: 10px;
    }
    .stProgress > div > div { background: #10a37f; }
    hr { border-color: #3d3d3d; }
    label { color: #c5c5d2 !important; font-size: 13px !important; }
    h1, h2, h3 { color: #ececec !important; }
    .stCaption { color: #8e8ea0 !important; }
    .stInfo { background: #1a2f2a !important; color: #4ade80 !important; border-color: #166534 !important; }
    .stWarning { background: #2f2a0a !important; color: #facc15 !important; border-color: #713f12 !important; }
    [data-testid="stMarkdownContainer"] p { color: #ececec; }
</style>
""", unsafe_allow_html=True)


# --- Modell laden ---

@st.cache_resource
def load_model():
    with open('model_data.pkl', 'rb') as f:
        return pickle.load(f)

data         = load_model()
model        = data['model']
le_target    = data['le_target']
le_dict      = data['le_dict']
feature_cols = data['feature_cols']
num_cols     = data['num_cols']
cat_cols     = data['cat_cols']
stats        = data['stats']
cat_values   = data['cat_values']

RISIKOGEWICHTE = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

# Label-Stufen für vereinfachte Slider (Low → Very High)
STUFEN        = ['Low', 'Medium', 'High', 'Very High']
STUFEN_ANZAHL = 4

def stufe_zu_wert(stufe_idx: int, col: str) -> float:
    """Mappt Stufe 0-3 auf den numerischen Wertebereich der Feature-Spalte."""
    s   = stats[col]
    pct = stufe_idx / (STUFEN_ANZAHL - 1)
    return s['min'] + pct * (s['max'] - s['min'])

# Features mit Label-Slidern, zugeordnet zu Scoring-Kriterien (Karrenbauer & Breitner 2022)
# Format: col -> (Label, Scoring-Kategorie, invertiert)
# invertiert=True: niedriger Wert = hohes Risiko (z. B. Resource_Availability)
FEATURES = {
    'Complexity_Score':               ('Complexity',           'Complexity', False),
    'Integration_Complexity':         ('Integration',          'Complexity', False),
    'Team_Size':                      ('Team Size',            'Complexity', False),
    'Project_Budget_USD':             ('Budget',               'Efficiency', False),
    'Budget_Utilization_Rate':        ('Budget Utilization',   'Efficiency', False),
    'Resource_Availability':          ('Resource Availability','Risk',       True),
    'Technical_Debt_Level':           ('Technical Debt',       'Risk',       False),
    'Team_Turnover_Rate':             ('Team Turnover',        'Risk',       False),
    'Previous_Delivery_Success_Rate': ('Past Delivery Success','Strategy',   True),
    'Schedule_Pressure':              ('Schedule Pressure',    'Urgency',    False),
}

# Typ A: Portfoliosummen (angelehnt an Eq. 4, Karrenbauer & Breitner 2022)
TYP_A = {
    'Team_Size':          ('Total Team Size',    100),
    'Project_Budget_USD': ('Total Budget (USD)', 10_000_000),
}

# Typ B: Portfoliodurchschnitte (angelehnt an Eq. 4, Karrenbauer & Breitner 2022)
TYP_B = {
    'Resource_Availability':   ('Avg. Resource Availability', 0.5,  True),
    'Budget_Utilization_Rate': ('Avg. Budget Utilization',    1.0,  False),
    'Schedule_Pressure':       ('Avg. Schedule Pressure',     0.15, False),
}


# --- Hilfsfunktionen ---

def risk_badge(level: str) -> str:
    return f'<span class="risk-{level.lower()}">{level}</span>'

def median_stufe(col: str) -> int:
    s      = stats[col]
    median = s['median']
    pct    = (median - s['min']) / max(s['max'] - s['min'], 1e-9)
    return int(round(pct * (STUFEN_ANZAHL - 1)))

def predict_project(eingabe: dict) -> tuple:
    row = {}
    for col in num_cols:
        row[col] = eingabe.get(col, stats[col]['median'])
    for col in cat_cols:
        raw = eingabe.get(col, cat_values[col][0])
        le  = le_dict[col]
        row[col] = int(le.transform([raw])[0]) if raw in le.classes_ else 0
    X_in       = pd.DataFrame([row])[feature_cols]
    proba      = model.predict_proba(X_in)[0]
    klasse     = le_target.inverse_transform([np.argmax(proba)])[0]
    return klasse, dict(zip(le_target.classes_, proba))

def get_portfolio_score(klassen: list) -> tuple:
    score = float(np.mean([RISIKOGEWICHTE[k] for k in klassen]))
    if score <= 1.5:   return score, 'Low'
    elif score <= 2.5: return score, 'Medium'
    elif score <= 3.5: return score, 'High'
    else:              return score, 'Critical'

def check_restriktionen(projekte: list) -> list:
    out = []
    for col, (label, limit) in TYP_A.items():
        summe = sum(p.get(col, 0) for p in projekte)
        if summe > limit:
            out.append(f"[Sum] {label}: {summe:,.0f} > {limit:,.0f}")
    for col, (label, schwelle, inv) in TYP_B.items():
        m = float(np.mean([p.get(col, stats[col]['median']) for p in projekte]))
        if (m < schwelle if inv else m > schwelle):
            out.append(f"[Avg] {label}: {m:.3f} ({'below' if inv else 'above'} {schwelle})")
    return out


# --- Session State ---

if 'portfolio'    not in st.session_state: st.session_state.portfolio    = []
if 'show_detail'  not in st.session_state: st.session_state.show_detail  = None
if 'numeric_mode' not in st.session_state: st.session_state.numeric_mode = False


# =====================================================================
# LAYOUT
# =====================================================================

st.markdown(
    "<h1 style='text-align:center; font-size:28px; margin-bottom:4px;'>"
    "IT Portfolio Risk Assessment</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#8e8ea0; margin-bottom:32px; font-size:13px;'>"
    "Prototyp · Aufbauend auf Karrenbauer & Breitner (2022)</p>",
    unsafe_allow_html=True
)

# --- Eingabebereich zentriert ---
_, mid, _ = st.columns([1, 3, 1])

with mid:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    projekt_name = st.text_input(
        "Project Name",
        placeholder="e.g. ERP Migration",
        key="pname",
        label_visibility="collapsed"
    )
    st.markdown(
        "<p style='color:#8e8ea0; font-size:12px; margin-bottom:16px;'>"
        "Enter a project name and adjust the parameters below.</p>",
        unsafe_allow_html=True
    )

    # Toggle: Label-Stufen vs. Zahlenwerte
    numeric_mode = st.toggle(
        "Show numeric values", value=st.session_state.numeric_mode, key="num_toggle"
    )
    st.session_state.numeric_mode = numeric_mode

    st.markdown("---")

    eingabe      = {}
    stufen_cache = {}

    # Scoring-Kategorien als Gruppen
    kategorien = {}
    for col, (label, kat, inv) in FEATURES.items():
        kategorien.setdefault(kat, []).append((col, label, inv))

    for kat, feats in kategorien.items():
        st.markdown(
            f"<p style='font-size:11px; color:#8e8ea0; letter-spacing:0.08em; "
            f"text-transform:uppercase; margin-bottom:8px;'>{kat}</p>",
            unsafe_allow_html=True
        )
        for col, label, inv in feats:
            default_stufe = median_stufe(col)
            if not numeric_mode:
                stufe = st.select_slider(
                    label,
                    options=STUFEN,
                    value=STUFEN[default_stufe],
                    key=f"s_{col}"
                )
                stufen_idx     = STUFEN.index(stufe)
                eingabe[col]   = stufe_zu_wert(stufen_idx, col)
                stufen_cache[col] = stufe
            else:
                s = stats[col]
                eingabe[col] = st.slider(
                    label,
                    min_value=float(s['min']),
                    max_value=float(s['max']),
                    value=float(s['median']),
                    step=(s['max'] - s['min']) / 100,
                    format="$%,.0f" if 'USD' in col else "%.3f",
                    key=f"n_{col}"
                )
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    b1, b2 = st.columns([3, 1])
    with b1:
        add_btn = st.button("＋ Add to Portfolio", use_container_width=True, type="primary")
    with b2:
        clear_btn = st.button("Clear", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if add_btn:
        name = projekt_name.strip() or f"Project {len(st.session_state.portfolio) + 1}"
        klasse, proba = predict_project(eingabe)
        st.session_state.portfolio.append({
            'name':   name,
            'klasse': klasse,
            'proba':  proba,
            'daten':  eingabe.copy(),
            'stufen': stufen_cache.copy() if not numeric_mode else {}
        })
        st.rerun()

    if clear_btn:
        st.session_state.portfolio   = []
        st.session_state.show_detail = None
        st.rerun()


# --- Portfolio-Ergebnisse ---

if not st.session_state.portfolio:
    st.markdown(
        "<p style='text-align:center; color:#8e8ea0; margin-top:40px;'>"
        "Add a project above to start the risk assessment.</p>",
        unsafe_allow_html=True
    )
    st.stop()

portfolio       = st.session_state.portfolio
klassen         = [p['klasse'] for p in portfolio]
score, g_klasse = get_portfolio_score(klassen)
verstösse       = check_restriktionen([p['daten'] for p in portfolio])

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown("### Portfolio Overview")

m1, m2, m3, m4 = st.columns(4)
for col_obj, title, value in [
    (m1, "Projects",           f"{len(portfolio)}"),
    (m2, "Avg. Risk Score",    f"{score:.2f} / 4"),
    (m3, "Portfolio Risk",     risk_badge(g_klasse)),
    (m4, "Restrictions",       f"{'⚠️' if verstösse else '✅'} {len(verstösse)}")
]:
    with col_obj:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# Projektliste
st.markdown("### Projects")

for i, proj in enumerate(portfolio):
    with st.container():
        c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
        with c1:
            st.markdown(f"**{proj['name']}**")
        with c2:
            st.markdown(risk_badge(proj['klasse']), unsafe_allow_html=True)
        with c3:
            p = proj['proba']
            st.caption(
                f"L {p.get('Low',0):.0%} · "
                f"M {p.get('Medium',0):.0%} · "
                f"H {p.get('High',0):.0%} · "
                f"C {p.get('Critical',0):.0%}"
            )
        with c4:
            if st.button("···", key=f"det_{i}"):
                st.session_state.show_detail = (
                    None if st.session_state.show_detail == i else i
                )

        if st.session_state.show_detail == i:
            with st.expander("", expanded=True):
                st.markdown(f"**Risk Probabilities – {proj['name']}**")
                proba_df = pd.DataFrame({
                    'Risk Level':    list(proj['proba'].keys()),
                    'Probability':   list(proj['proba'].values())
                }).sort_values('Probability', ascending=False)
                st.bar_chart(proba_df.set_index('Risk Level'), color="#10a37f")

                if proj.get('stufen'):
                    st.markdown("**Input Parameters:**")
                    rows = [
                        (FEATURES[k][0], v, FEATURES[k][2])
                        for k, v in proj['stufen'].items()
                        if k in FEATURES
                    ]
                    st.dataframe(
                        pd.DataFrame(rows, columns=['Feature', 'Level', 'Inverted'
                                                    ])[['Feature', 'Level']],
                        hide_index=True, use_container_width=True
                    )
        st.divider()

# Restriktions-Tracking
st.markdown("### Restriction Tracking")
st.caption("Based on optimization constraints from Karrenbauer & Breitner (2022), Eq. 3–8")

r1, r2 = st.columns(2)

with r1:
    st.markdown("**Type A – Portfolio Sums**")
    for col, (label, limit) in TYP_A.items():
        summe = sum(p['daten'].get(col, 0) for p in portfolio)
        icon  = '✅' if summe <= limit else '⚠️'
        st.markdown(f"{icon} **{label}** `{summe:,.0f}` / `{limit:,.0f}`")
        st.progress(min(summe / limit, 1.0))

with r2:
    st.markdown("**Type B – Portfolio Averages**")
    for col, (label, schwelle, inv) in TYP_B.items():
        m     = float(np.mean([p['daten'].get(col, stats[col]['median']) for p in portfolio]))
        ok    = m >= schwelle if inv else m <= schwelle
        icon  = '✅' if ok else '⚠️'
        pct   = float(np.clip(m / schwelle if not inv else schwelle / max(m, 1e-9), 0, 1))
        st.markdown(f"{icon} **{label}** `{m:.3f}` / `{schwelle}`")
        st.progress(pct)

# Risikoverteilung
st.markdown("### Risk Distribution")
verteilung = pd.Series(klassen).value_counts().reindex(
    ['Low', 'Medium', 'High', 'Critical'], fill_value=0
)
st.bar_chart(verteilung, color="#10a37f")

# Handlungsempfehlung
st.markdown("### Recommendation")
st.info({
    'Low':      '✅ Portfolio well-positioned. Regular monitoring is sufficient.',
    'Medium':   '🟡 Portfolio acceptable. Monitor high-risk projects closely.',
    'High':     '🔴 Portfolio critical. Reprioritize resources.',
    'Critical': '🚨 Portfolio unsustainable. Immediate restructuring required.'
}[g_klasse])

if verstösse:
    st.warning("**Restriction Violations:**\n" + "\n".join(f"- {v}" for v in verstösse))

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.caption(
    "This tool is a scientific prototype. "
    "Final portfolio decisions remain with experienced executives "
    "(Karrenbauer & Breitner 2022)."
)
