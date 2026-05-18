"""
IT Portfolio Risk Assessment
Prototyp zur Vorhersage von IT-Projektrisikoniveaus mittels ML
Aufbauend auf: Karrenbauer & Breitner (2022)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

st.set_page_config(
    page_title="IT Portfolio Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .stApp { background:#212121; color:#ececec; }
  section[data-testid="stSidebar"] > div:first-child {
      background:#171717; border-right:1px solid #2d2d2d;
  }

  /* Input-Bar */
  .input-bar {
      display:flex; align-items:center; gap:10px;
      background:#2f2f2f; border:1px solid #404040;
      border-radius:999px; padding:10px 18px;
      margin-bottom:32px;
  }
  .input-bar input { background:transparent; border:none; outline:none;
      color:#ececec; font-size:15px; flex:1; }

  /* Karten */
  .card {
      background:#2f2f2f; border:1px solid #3a3a3a;
      border-radius:14px; padding:20px 22px; margin-bottom:12px;
  }
  .card-sm {
      background:#2a2a2a; border:1px solid #3a3a3a;
      border-radius:10px; padding:12px 16px; margin-bottom:8px;
  }

  /* Metriken */
  .metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
  .metric-box {
      background:#2f2f2f; border:1px solid #3a3a3a; border-radius:14px;
      padding:18px 16px; text-align:center;
  }
  .metric-title { font-size:11px; color:#8e8ea0; letter-spacing:.06em;
      text-transform:uppercase; margin-bottom:8px; }
  .metric-value { font-size:22px; font-weight:700; color:#ececec; }

  /* Risk-Badges */
  .badge-low      { background:#0d2a1a; color:#4ade80; border:1px solid #14532d;
      padding:4px 12px; border-radius:999px; font-weight:600; font-size:13px; display:inline-block; }
  .badge-medium   { background:#2d2008; color:#fbbf24; border:1px solid #78350f;
      padding:4px 12px; border-radius:999px; font-weight:600; font-size:13px; display:inline-block; }
  .badge-high     { background:#2d0d0d; color:#f87171; border:1px solid #7f1d1d;
      padding:4px 12px; border-radius:999px; font-weight:600; font-size:13px; display:inline-block; }
  .badge-critical { background:#3d0808; color:#fca5a5; border:1px solid #991b1b;
      padding:4px 12px; border-radius:999px; font-weight:600; font-size:13px; display:inline-block; }

  /* Sidebar-Einträge */
  .sidebar-item {
      background:#1e1e1e; border:1px solid #2d2d2d; border-radius:10px;
      padding:10px 14px; margin-bottom:6px; cursor:pointer;
  }
  .sidebar-item:hover { background:#252525; border-color:#404040; }
  .sidebar-item-title { font-size:13px; font-weight:600; color:#ececec; }
  .sidebar-item-meta  { font-size:11px; color:#8e8ea0; margin-top:2px; }

  /* Pending-Projekt-Tags */
  .proj-tag {
      display:inline-block; background:#1a3a2a; color:#4ade80;
      border:1px solid #14532d; border-radius:999px;
      padding:3px 12px; font-size:12px; margin:3px;
  }

  /* Kategorie-Label */
  .cat-label {
      font-size:10px; color:#8e8ea0; letter-spacing:.1em;
      text-transform:uppercase; margin:20px 0 8px;
      border-bottom:1px solid #2d2d2d; padding-bottom:6px;
  }

  /* Allgemeine Anpassungen */
  .stButton > button {
      border-radius:10px !important; font-weight:600 !important;
      border:1px solid #3a3a3a !important;
  }
  .stButton > button[kind="primary"] {
      background:#10a37f !important; border-color:#10a37f !important; color:#fff !important;
  }
  .stButton > button[kind="secondary"] {
      background:#2f2f2f !important; color:#ececec !important;
  }
  div[data-testid="stSelectSlider"] label { color:#c5c5d2 !important; font-size:13px !important; }
  div[data-testid="stSelectSlider"] div[data-baseweb="slider"] { margin-top:4px; }
  .stProgress > div > div > div { background:#10a37f !important; border-radius:999px; }
  .stDivider { border-color:#2d2d2d !important; }
  h1,h2,h3 { color:#ececec !important; }
  p, li { color:#c5c5d2; }
  .stCaption { color:#8e8ea0 !important; }
  .stInfo    { background:#0d2a1a !important; color:#4ade80 !important; border-color:#14532d !important; border-radius:12px !important; }
  .stWarning { background:#2d2008 !important; color:#fbbf24 !important; border-color:#78350f !important; border-radius:12px !important; }
  .stSuccess { background:#0d2a1a !important; color:#4ade80 !important; border-color:#14532d !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Modell laden ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_data.pkl', 'rb') as f:
        return pickle.load(f)

md           = load_model()
model        = md['model']
le_target    = md['le_target']
le_dict      = md['le_dict']
feature_cols = md['feature_cols']
num_cols     = md['num_cols']
cat_cols     = md['cat_cols']
stats        = md['stats']
cat_values   = md['cat_values']

RISIKOGEWICHTE = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

# ── Smarte Slider-Definitionen ────────────────────────────────────────────────
# Für numerisch messbare Kategorien: konkrete Wertebereiche
# Für qualitative Kategorien: Low / Medium / High / Very High
# N/A = Standardwert (Median), wird beim Predict durch Median ersetzt

SLIDERS = {
    # ── Complexity ────────────────────────────────────────────────────────────
    'Team_Size': dict(
        label='Team Size', cat='Complexity',
        options=['N/A', '1–5', '6–10', '11–20', '20+'],
        values =[None,    3,    8,      15,      24]
    ),
    'Complexity_Score': dict(
        label='Complexity', cat='Complexity',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   3.0,   5.5,     7.5,    9.5]
    ),
    'Integration_Complexity': dict(
        label='Integration Complexity', cat='Complexity',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   2.0,   4.5,     7.0,    9.5]
    ),
    # ── Efficiency ───────────────────────────────────────────────────────────
    'Project_Budget_USD': dict(
        label='Budget', cat='Efficiency',
        options=['N/A', '< $500k', '$500k–$1M', '$1M–$1.5M', '> $1.5M'],
        values =[None,  350_000,   750_000,     1_250_000,   1_800_000]
    ),
    'Budget_Utilization_Rate': dict(
        label='Budget Utilization', cat='Efficiency',
        options=['N/A', '< 75 %', '75–90 %', '90–110 %', '> 110 %'],
        values =[None,   0.65,     0.82,       1.00,       1.25]
    ),
    # ── Risk ─────────────────────────────────────────────────────────────────
    'Resource_Availability': dict(
        label='Resource Availability', cat='Risk',
        options=['N/A', 'Very Low', 'Low', 'Medium', 'High'],
        values =[None,   0.35,       0.50,  0.65,    0.90]
    ),
    'Technical_Debt_Level': dict(
        label='Technical Debt', cat='Risk',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   0.15,  0.40,    0.65,   0.90]
    ),
    'Team_Turnover_Rate': dict(
        label='Team Turnover', cat='Risk',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   0.05,  0.20,    0.45,   0.75]
    ),
    # ── Strategy ─────────────────────────────────────────────────────────────
    'Previous_Delivery_Success_Rate': dict(
        label='Past Delivery Success', cat='Strategy',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   0.25,  0.55,    0.80,   0.95]
    ),
    # ── Urgency ──────────────────────────────────────────────────────────────
    'Schedule_Pressure': dict(
        label='Schedule Pressure', cat='Urgency',
        options=['N/A', 'Low', 'Medium', 'High', 'Very High'],
        values =[None,   0.03,  0.12,    0.28,   0.45]
    ),
}

# Typ A: Portfoliosummen (Karrenbauer & Breitner 2022, Eq. 4)
TYP_A = {
    'Team_Size':          ('Total Team Size',    100),
    'Project_Budget_USD': ('Total Budget (USD)', 10_000_000),
}
# Typ B: Portfoliodurchschnitte (Karrenbauer & Breitner 2022, Eq. 4)
TYP_B = {
    'Resource_Availability':   ('Avg. Resource Avail.', 0.5,  True),
    'Budget_Utilization_Rate': ('Avg. Budget Util.',    1.0,  False),
    'Schedule_Pressure':       ('Avg. Schedule Press.', 0.15, False),
}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def badge(level: str) -> str:
    return f'<span class="badge-{level.lower()}">{level}</span>'

def opt_to_val(col: str, option: str) -> float:
    """Wandelt Slider-Option in numerischen Wert um. N/A → Median."""
    d = SLIDERS[col]
    idx = d['options'].index(option)
    v   = d['values'][idx]
    return stats[col]['median'] if v is None else float(v)

def predict(eingabe: dict) -> tuple:
    row = {}
    for col in num_cols:
        row[col] = float(eingabe.get(col, stats[col]['median']))
    for col in cat_cols:
        raw = eingabe.get(col, cat_values[col][0])
        le  = le_dict[col]
        row[col] = int(le.transform([raw])[0]) if raw in le.classes_ else 0
    X   = pd.DataFrame([row])[feature_cols]
    p   = model.predict_proba(X)[0]
    k   = le_target.inverse_transform([np.argmax(p)])[0]
    return k, dict(zip(le_target.classes_, p))

def portfolio_risk(klassen: list) -> tuple:
    s = float(np.mean([RISIKOGEWICHTE[k] for k in klassen]))
    if s <= 1.5: return s, 'Low'
    if s <= 2.5: return s, 'Medium'
    if s <= 3.5: return s, 'High'
    return s, 'Critical'

def check_restrictions(projekte: list) -> list:
    out = []
    for col, (lbl, lim) in TYP_A.items():
        sm = sum(p.get(col, 0) for p in projekte)
        if sm > lim:
            out.append(f"[Sum] {lbl}: {sm:,.0f} > {lim:,.0f}")
    for col, (lbl, thr, inv) in TYP_B.items():
        m = float(np.mean([p.get(col, stats[col]['median']) for p in projekte]))
        if (m < thr if inv else m > thr):
            out.append(f"[Avg] {lbl}: {m:.3f} ({'below' if inv else 'above'} {thr})")
    return out

RECOMMENDATIONS = {
    'Low':      '✅ Portfolio well-positioned. Regular monitoring is sufficient.',
    'Medium':   '🟡 Portfolio acceptable. Monitor high-risk projects closely.',
    'High':     '🔴 Portfolio critical. Reprioritize resources immediately.',
    'Critical': '🚨 Portfolio unsustainable. Immediate restructuring required.',
}


# ── Session State initialisieren ──────────────────────────────────────────────
defaults = {
    'view':            'input',   # 'input' | 'results'
    'pending':         [],        # Projekte im aktuellen Aufbau
    'saved':           [],        # Gespeicherte abgeschlossene Portfolios
    'active_saved':    None,      # Index des angezeigten gespeicherten Portfolios
    'results':         None,      # Berechnetes Ergebnis-Portfolio
    'slider_vals':     {},        # Aktuelle Slider-Auswahlen
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR: Portfolio-Verlauf ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:18px; font-weight:700; color:#ececec; "
        "margin-bottom:16px;'>📊 IT Risk Assessment</p>",
        unsafe_allow_html=True
    )

    if st.button("＋  New Portfolio", use_container_width=True, type="primary"):
        st.session_state.view         = 'input'
        st.session_state.pending      = []
        st.session_state.results      = None
        st.session_state.slider_vals  = {}
        st.session_state.active_saved = None
        st.rerun()

    st.markdown("<div style='margin:12px 0 8px; font-size:11px; color:#8e8ea0; "
                "letter-spacing:.08em; text-transform:uppercase;'>Saved Portfolios</div>",
                unsafe_allow_html=True)

    if not st.session_state.saved:
        st.markdown("<p style='font-size:12px; color:#555;'>No portfolios saved yet.</p>",
                    unsafe_allow_html=True)
    else:
        for i, pf in enumerate(st.session_state.saved):
            col_a, col_b = st.columns([5, 1])
            with col_a:
                if st.button(
                    f"{'▶ ' if st.session_state.active_saved == i else ''}"
                    f"{pf['name']}",
                    key=f"load_{i}", use_container_width=True
                ):
                    st.session_state.active_saved = i
                    st.session_state.results      = pf
                    st.session_state.view         = 'results'
                    st.rerun()
            with col_b:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.saved.pop(i)
                    if st.session_state.active_saved == i:
                        st.session_state.active_saved = None
                        st.session_state.view         = 'input'
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == 'input':

    st.markdown(
        "<h1 style='text-align:center; font-size:26px; "
        "margin:8px 0 4px;'>IT Portfolio Risk Assessment</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#8e8ea0; font-size:13px; "
        "margin-bottom:28px;'>Aufbauend auf Karrenbauer & Breitner (2022)</p>",
        unsafe_allow_html=True
    )

    # ── Eingabe-Bar (zentriert, ChatGPT-Stil) ─────────────────────────────────
    _, center, _ = st.columns([1, 4, 1])
    with center:
        bar_l, bar_r = st.columns([6, 1])
        with bar_l:
            proj_name = st.text_input(
                "pname", placeholder="Project name…",
                label_visibility="collapsed", key="proj_name_input"
            )
        with bar_r:
            submit_btn = st.button("→", type="primary", use_container_width=True)

        # Pending-Projekte anzeigen
        if st.session_state.pending:
            tags = " ".join(
                f'<span class="proj-tag">✓ {p["name"]}</span>'
                for p in st.session_state.pending
            )
            st.markdown(
                f"<div style='margin:8px 0 0;'>{tags}</div>",
                unsafe_allow_html=True
            )
            st.caption(
                f"{len(st.session_state.pending)} project(s) added. "
                "Fill sliders for next project or hit → to calculate."
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Slider-Bereich ────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)

        kategorien = {}
        for col, d in SLIDERS.items():
            kategorien.setdefault(d['cat'], []).append(col)

        eingabe = {}
        for kat, cols_ in kategorien.items():
            st.markdown(f'<div class="cat-label">{kat}</div>', unsafe_allow_html=True)
            for col in cols_:
                d       = SLIDERS[col]
                default = st.session_state.slider_vals.get(col, 'N/A')
                if default not in d['options']:
                    default = 'N/A'
                chosen = st.select_slider(
                    d['label'],
                    options=d['options'],
                    value=default,
                    key=f"sl_{col}"
                )
                st.session_state.slider_vals[col] = chosen
                eingabe[col] = opt_to_val(col, chosen)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Projekt hinzufügen ────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("＋  Add another project", use_container_width=True):
            name = proj_name.strip() or f"Project {len(st.session_state.pending) + 1}"
            st.session_state.pending.append({
                'name':  name,
                'daten': eingabe.copy(),
                'opts':  {c: st.session_state.slider_vals.get(c, 'N/A') for c in SLIDERS}
            })
            # Slider zurücksetzen
            for col in SLIDERS:
                st.session_state.slider_vals[col] = 'N/A'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit: Ergebnis berechnen ────────────────────────────────────────────
    if submit_btn:
        name  = proj_name.strip() or f"Project {len(st.session_state.pending) + 1}"
        k, p  = predict(eingabe)
        alle  = st.session_state.pending + [{
            'name': name, 'daten': eingabe.copy(),
            'opts': {c: st.session_state.slider_vals.get(c, 'N/A') for c in SLIDERS}
        }]
        # Für jedes Projekt Vorhersage berechnen
        for proj in alle:
            if 'klasse' not in proj:
                proj['klasse'], proj['proba'] = predict(proj['daten'])

        ts    = datetime.now().strftime("%d.%m. %H:%M")
        score, g_k = portfolio_risk([pr['klasse'] for pr in alle])
        result = {
            'name':       f"Portfolio {len(st.session_state.saved) + 1}  ·  {ts}",
            'projects':   alle,
            'score':      score,
            'klasse':     g_k,
            'violations': check_restrictions([pr['daten'] for pr in alle]),
            'ts':         ts,
        }
        st.session_state.results      = result
        st.session_state.view         = 'results'
        st.session_state.pending      = []
        st.session_state.slider_vals  = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.view == 'results':
    res = st.session_state.results
    if res is None:
        st.session_state.view = 'input'
        st.rerun()

    projects   = res['projects']
    klassen    = [p['klasse'] for p in projects]
    score      = res['score']
    g_klasse   = res['klasse']
    violations = res['violations']

    # ── Top-Bar ───────────────────────────────────────────────────────────────
    top_l, top_r = st.columns([1, 1])
    with top_l:
        if st.button("← Back", type="secondary"):
            st.session_state.view        = 'input'
            st.session_state.results     = None
            st.session_state.pending     = []
            st.session_state.slider_vals = {}
            st.rerun()
    with top_r:
        if st.button("💾  Save Portfolio", type="primary"):
            already = any(s['name'] == res['name'] for s in st.session_state.saved)
            if not already:
                st.session_state.saved.append(res)
                st.session_state.active_saved = len(st.session_state.saved) - 1
            st.success("Portfolio saved.")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"### {res['name']}")

    # ── Kennzahlen ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col_obj, title, value in [
        (m1, "Projects",         str(len(projects))),
        (m2, "Avg. Risk Score",  f"{score:.2f} / 4.00"),
        (m3, "Portfolio Risk",   badge(g_klasse)),
        (m4, "Restrictions",     f"{'⚠️' if violations else '✅'}  {len(violations)}"),
    ]:
        with col_obj:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Projektliste ──────────────────────────────────────────────────────────
    st.markdown("#### Projects")

    for i, proj in enumerate(projects):
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
            with c1:
                st.markdown(f"**{proj['name']}**")
            with c2:
                st.markdown(badge(proj['klasse']), unsafe_allow_html=True)
            with c3:
                p = proj['proba']
                st.caption(
                    f"L {p.get('Low',0):.0%}  ·  "
                    f"M {p.get('Medium',0):.0%}  ·  "
                    f"H {p.get('High',0):.0%}  ·  "
                    f"C {p.get('Critical',0):.0%}"
                )
            with c4:
                with st.expander("···"):
                    st.markdown(f"**Probabilities – {proj['name']}**")
                    pdf = pd.DataFrame({
                        'Risk':  list(proj['proba'].keys()),
                        'Prob':  list(proj['proba'].values())
                    }).sort_values('Prob', ascending=False)
                    st.bar_chart(pdf.set_index('Risk'), color="#10a37f")

                    if proj.get('opts'):
                        st.markdown("**Inputs:**")
                        rows = [
                            (SLIDERS[k]['label'], v)
                            for k, v in proj['opts'].items()
                            if v != 'N/A'
                        ]
                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows, columns=['Feature', 'Level']),
                                hide_index=True, use_container_width=True
                            )
                        else:
                            st.caption("All sliders set to N/A.")
            st.divider()

    # ── Risikoverteilung ──────────────────────────────────────────────────────
    st.markdown("#### Risk Distribution")
    dist = pd.Series(klassen).value_counts().reindex(
        ['Low', 'Medium', 'High', 'Critical'], fill_value=0
    )
    st.bar_chart(dist, color="#10a37f")

    # ── Restriktions-Tracking ─────────────────────────────────────────────────
    st.markdown("#### Restriction Tracking")
    st.caption("Based on Karrenbauer & Breitner (2022), Eq. 3–8")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Type A – Sums**")
        for col, (lbl, lim) in TYP_A.items():
            sm = sum(p['daten'].get(col, 0) for p in projects)
            st.markdown(
                f"{'✅' if sm <= lim else '⚠️'} **{lbl}**  "
                f"`{sm:,.0f}` / `{lim:,.0f}`"
            )
            st.progress(min(sm / lim, 1.0))

    with r2:
        st.markdown("**Type B – Averages**")
        for col, (lbl, thr, inv) in TYP_B.items():
            m  = float(np.mean([p['daten'].get(col, stats[col]['median']) for p in projects]))
            ok = m >= thr if inv else m <= thr
            pct = float(np.clip(m / thr if not inv else thr / max(m, 1e-9), 0, 1))
            st.markdown(
                f"{'✅' if ok else '⚠️'} **{lbl}**  "
                f"`{m:.3f}` / `{thr}`"
            )
            st.progress(pct)

    # ── Handlungsempfehlung ───────────────────────────────────────────────────
    st.markdown("#### Recommendation")
    st.info(RECOMMENDATIONS[g_klasse])

    if violations:
        st.warning("**Restriction Violations:**\n" + "\n".join(f"- {v}" for v in violations))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption(
        "Scientific prototype · Final decisions remain with experienced executives "
        "(Karrenbauer & Breitner 2022)"
    )
