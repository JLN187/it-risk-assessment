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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .risk-low      { background:#d4edda; color:#155724; padding:6px 14px;
                     border-radius:20px; font-weight:600; display:inline-block; }
    .risk-medium   { background:#fff3cd; color:#856404; padding:6px 14px;
                     border-radius:20px; font-weight:600; display:inline-block; }
    .risk-high     { background:#f8d7da; color:#721c24; padding:6px 14px;
                     border-radius:20px; font-weight:600; display:inline-block; }
    .risk-critical { background:#842029; color:#ffffff; padding:6px 14px;
                     border-radius:20px; font-weight:600; display:inline-block; }
    .metric-box    { background:#f8f9fa; border-radius:10px; padding:16px;
                     text-align:center; border:1px solid #dee2e6; }
    .metric-title  { font-size:13px; color:#6c757d; margin-bottom:4px; }
    .metric-value  { font-size:26px; font-weight:700; color:#212529; }
</style>
""", unsafe_allow_html=True)


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

# Features fuer Slider-Eingabe, zugeordnet zu Scoring-Kriterien (Karrenbauer & Breitner 2022)
SLIDER_FEATURES = {
    'Complexity_Score':               ('Complexity Score',        'Complexity'),
    'Integration_Complexity':         ('Integration Complexity',  'Complexity'),
    'Team_Size':                      ('Team Size',               'Complexity'),
    'Project_Budget_USD':             ('Budget (USD)',            'Efficiency'),
    'Budget_Utilization_Rate':        ('Budget Utilization Rate', 'Efficiency'),
    'Resource_Availability':          ('Resource Availability',   'Risk'),
    'Technical_Debt_Level':           ('Technical Debt Level',    'Risk'),
    'Team_Turnover_Rate':             ('Team Turnover Rate',      'Risk'),
    'Previous_Delivery_Success_Rate': ('Prev. Delivery Success',  'Strategy'),
    'Organizational_Change_Frequency':('Org. Change Frequency',   'Strategy'),
    'Regulatory_Compliance_Level':    ('Regulatory Compliance',   'Urgency'),
    'Schedule_Pressure':              ('Schedule Pressure',       'Urgency'),
}

DROPDOWN_FEATURES = {
    'Methodology_Used':           'Vorgehensmodell',
    'Org_Process_Maturity':       'Prozessreife',
    'Tech_Environment_Stability': 'Tech-Stabilitaet',
    'Team_Experience_Level':      'Team-Erfahrung',
}

# Typ A: Portfoliosummen (angelehnt an Eq. 4, Karrenbauer & Breitner 2022)
TYP_A = {
    'Team_Size':          ('Gesamtteamgroesse',   100),
    'Project_Budget_USD': ('Gesamtbudget (USD)',  10_000_000),
}

# Typ B: Portfoliodurchschnitte (angelehnt an Eq. 4, Karrenbauer & Breitner 2022)
TYP_B = {
    'Resource_Availability':   ('Ressourcenverfuegb. (Ø)', 0.5,  True),
    'Budget_Utilization_Rate': ('Budgetauslastung (Ø)',    1.0,  False),
    'Schedule_Pressure':       ('Zeitdruck (Ø)',           0.15, False),
}


def risk_badge(level):
    return f'<span class="risk-{level.lower()}">{level}</span>'


def predict_project(eingabe):
    row = {}
    for col in num_cols:
        row[col] = eingabe.get(col, stats[col]['median'])
    for col in cat_cols:
        raw = eingabe.get(col, cat_values[col][0])
        le  = le_dict[col]
        row[col] = int(le.transform([raw])[0]) if raw in le.classes_ else 0
    X_input    = pd.DataFrame([row])[feature_cols]
    proba      = model.predict_proba(X_input)[0]
    klasse     = le_target.inverse_transform([np.argmax(proba)])[0]
    proba_dict = dict(zip(le_target.classes_, proba))
    return klasse, proba_dict


def get_portfolio_score(klassen):
    score = float(np.mean([RISIKOGEWICHTE[k] for k in klassen]))
    if score <= 1.5:   return score, 'Low'
    elif score <= 2.5: return score, 'Medium'
    elif score <= 3.5: return score, 'High'
    else:              return score, 'Critical'


def check_restriktionen(projekte):
    verstösse = []
    for col, (label, limit) in TYP_A.items():
        summe = sum(p.get(col, 0) for p in projekte)
        if summe > limit:
            verstösse.append(f"[Summe] {label}: {summe:,.0f} > Limit {limit:,.0f}")
    for col, (label, schwelle, inv) in TYP_B.items():
        mittel   = float(np.mean([p.get(col, stats[col]['median']) for p in projekte]))
        verletzt = mittel < schwelle if inv else mittel > schwelle
        if verletzt:
            verstösse.append(
                f"[Ø] {label}: {mittel:.3f} "
                f"{'unter' if inv else 'ueber'} Schwelle {schwelle}"
            )
    return verstösse


if 'portfolio' not in st.session_state:
    st.session_state.portfolio   = []
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = None


with st.sidebar:
    st.title("Projekt hinzufügen")
    st.caption("Aufbauend auf Karrenbauer & Breitner (2022)")
    st.divider()

    projekt_name = st.text_input("Projektname", placeholder="z. B. ERP Migration", key="pname")
    st.markdown("**Projektmerkmale** *(Scoring-Framework)*")

    eingabe = {}

    for col, (label, kategorie) in SLIDER_FEATURES.items():
        if col not in num_cols:
            continue
        s = stats[col]
        eingabe[col] = st.slider(
            label,
            min_value=float(s['min']),
            max_value=float(s['max']),
            value=float(s['median']),
            step=(s['max'] - s['min']) / 100,
            format="$%,.0f" if 'USD' in col else "%.2f",
            help=f"Scoring-Kategorie: {kategorie}"
        )

    st.divider()
    st.markdown("**Weitere Merkmale**")

    for col, label in DROPDOWN_FEATURES.items():
        if col in cat_cols and col in cat_values:
            eingabe[col] = st.selectbox(label, options=cat_values[col], index=0, key=f"dd_{col}")

    st.divider()
    b1, b2 = st.columns(2)

    with b1:
        add_btn = st.button("+ Zum Portfolio", use_container_width=True, type="secondary")
    with b2:
        clear_btn = st.button("Leeren", use_container_width=True)

    if add_btn:
        name = projekt_name.strip() or f"Projekt {len(st.session_state.portfolio) + 1}"
        klasse, proba = predict_project(eingabe)
        st.session_state.portfolio.append({
            'name': name, 'klasse': klasse,
            'proba': proba, 'daten': eingabe.copy()
        })
        st.success(f"{name} hinzugefügt – {klasse}")

    if clear_btn:
        st.session_state.portfolio   = []
        st.session_state.show_detail = None
        st.rerun()


st.title("IT Portfolio Risk Assessment")
st.caption("Prototyp · Aufbauend auf Karrenbauer & Breitner (2022)")

if not st.session_state.portfolio:
    st.info("Füge links ein Projekt zum Portfolio hinzu, um die Risikoauswertung zu starten.")
    st.stop()

portfolio       = st.session_state.portfolio
klassen         = [p['klasse'] for p in portfolio]
score, g_klasse = get_portfolio_score(klassen)
verstösse       = check_restriktionen([p['daten'] for p in portfolio])

st.markdown("### Portfolio-Übersicht")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="metric-box"><div class="metric-title">Projekte</div>'
        f'<div class="metric-value">{len(portfolio)}</div></div>',
        unsafe_allow_html=True)
with m2:
    st.markdown(
        f'<div class="metric-box"><div class="metric-title">Ø Risikowert</div>'
        f'<div class="metric-value">{score:.2f} / 4</div></div>',
        unsafe_allow_html=True)
with m3:
    st.markdown(
        f'<div class="metric-box"><div class="metric-title">Gesamtrisikoklasse</div>'
        f'<div class="metric-value">{risk_badge(g_klasse)}</div></div>',
        unsafe_allow_html=True)
with m4:
    farbe = "🔴" if verstösse else "🟢"
    st.markdown(
        f'<div class="metric-box"><div class="metric-title">Restriktionen</div>'
        f'<div class="metric-value">{farbe} {len(verstösse)}</div></div>',
        unsafe_allow_html=True)

st.divider()
st.markdown("### Projekte im Portfolio")

for i, proj in enumerate(portfolio):
    with st.container():
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            st.markdown(f"**{proj['name']}**")
        with c2:
            st.markdown(risk_badge(proj['klasse']), unsafe_allow_html=True)
        with c3:
            p = proj['proba']
            st.caption(
                f"L:{p.get('Low',0):.0%}  M:{p.get('Medium',0):.0%}  "
                f"H:{p.get('High',0):.0%}  C:{p.get('Critical',0):.0%}"
            )
        with c4:
            if st.button("Details", key=f"det_{i}"):
                st.session_state.show_detail = (
                    None if st.session_state.show_detail == i else i
                )

        if st.session_state.show_detail == i:
            with st.expander("", expanded=True):
                st.markdown(f"**Wahrscheinlichkeiten – {proj['name']}**")
                proba_df = pd.DataFrame({
                    'Risikoklasse':       list(proj['proba'].keys()),
                    'Wahrscheinlichkeit': list(proj['proba'].values())
                }).sort_values('Wahrscheinlichkeit', ascending=False)
                st.bar_chart(proba_df.set_index('Risikoklasse'), color="#4C72B0")

                st.markdown("**Eingabedaten:**")
                st.dataframe(
                    pd.DataFrame(
                        [(SLIDER_FEATURES[k][0], round(v, 3))
                         for k, v in proj['daten'].items() if k in SLIDER_FEATURES],
                        columns=['Merkmal', 'Wert']
                    ),
                    hide_index=True, use_container_width=True
                )
        st.divider()

st.markdown("### Restriktions-Tracking")
st.caption("Angelehnt an Karrenbauer & Breitner (2022), Eq. 3–8")

r1, r2 = st.columns(2)

with r1:
    st.markdown("**Typ A – Portfoliosummen**")
    for col, (label, limit) in TYP_A.items():
        summe = sum(p['daten'].get(col, 0) for p in portfolio)
        st.markdown(
            f"{'✅' if summe <= limit else '⚠️'} **{label}** "
            f"`{summe:,.0f}` / Limit `{limit:,.0f}`"
        )
        st.progress(min(summe / limit, 1.0))

with r2:
    st.markdown("**Typ B – Portfoliodurchschnitte**")
    for col, (label, schwelle, inv) in TYP_B.items():
        mittel   = float(np.mean([p['daten'].get(col, stats[col]['median']) for p in portfolio]))
        verletzt = mittel < schwelle if inv else mittel > schwelle
        pct      = min(mittel / schwelle if not inv else schwelle / max(mittel, 0.001), 1.0)
        st.markdown(
            f"{'✅' if not verletzt else '⚠️'} **{label}** "
            f"`{mittel:.3f}` / Schwelle `{schwelle}`"
        )
        st.progress(float(np.clip(pct, 0, 1)))

st.markdown("### Risikoverteilung im Portfolio")
st.bar_chart(
    pd.Series(klassen).value_counts().reindex(
        ['Low', 'Medium', 'High', 'Critical'], fill_value=0
    ),
    color="#4C72B0"
)

st.markdown("### Handlungsempfehlung")
st.info({
    'Low':      '✅ Portfolio gut aufgestellt. Regelmäßiges Monitoring ausreichend.',
    'Medium':   '🟡 Portfolio akzeptabel. Hochrisikoprojekte gezielt überwachen.',
    'High':     '🔴 Portfolio kritisch. Ressourcen neu priorisieren.',
    'Critical': '🚨 Portfolio nicht tragbar. Sofortige Portfolioanpassung nötig.'
}[g_klasse])

if verstösse:
    st.warning("**Restriktionsverletzungen:**\n" + "\n".join(f"- {v}" for v in verstösse))

st.caption(
    "Hinweis: Dieses Tool ist ein wissenschaftlicher Prototyp. "
    "Finale Portfolioentscheidungen obliegen immer erfahrenen Führungskräften "
    "(Karrenbauer & Breitner 2022)."
)
