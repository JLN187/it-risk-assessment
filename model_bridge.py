"""
model_bridge.py - Verbindung zwischen Oberfläche und trainiertem Modell.

Stellt bereit: Merkmalsschema mit lesbaren Stufen, Risikorichtung je Merkmal,
Einzelvorhersage, Portfolioaggregation und SHAP-basierte Erklärung.
Enthält bewusst keine Streamlit-Abhängigkeit, damit die Logik unabhängig von
der Oberfläche geprüft werden kann.
"""
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Merkmalsschema
# ---------------------------------------------------------------------------
ORDINAL_ORDERS = {
    "Requirement_Stability":       ["Volatile", "Moderate", "Stable"],
    "Regulatory_Compliance_Level": ["Low", "Medium", "High", "Critical"],
    "Technology_Familiarity":      ["New", "Familiar", "Expert"],
    "Stakeholder_Engagement_Level":["Poor", "Low", "Medium", "High", "Excellent"],
    "Executive_Sponsorship":       ["Weak", "Moderate", "Strong"],
    "Priority_Level":              ["Low", "Medium", "High", "Critical"],
    "Org_Process_Maturity":        ["Ad-hoc", "Defined", "Managed", "Optimizing"],
    "Data_Security_Requirements":  ["Low", "Medium", "High", "Strict"],
    "Key_Stakeholder_Availability":["Poor", "Limited", "Moderate", "Good", "Excellent"],
    "Tech_Environment_Stability":  ["Legacy/Unstable", "Mixed", "Modern/Stable"],
    "Resource_Contention_Level":   ["Low", "Medium", "High"],
    "Industry_Volatility":         ["Stable", "Moderate", "High", "Extreme"],
    "Documentation_Quality":       ["Poor", "Basic", "Good", "Excellent"],
    "Team_Experience_Level":       ["Junior", "Mixed", "Senior", "Expert"],
    "Project_Manager_Experience":  ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
    "Client_Experience_Level":     ["First-time", "Occasional", "Regular", "Strategic"],
}
# "None" bezeichnet das Fehlen eines Prozesses und bildet die unterste Rangstufe
MATURITY_ORDER = ["None", "Basic", "Formal", "Advanced"]
MATURITY_COLS  = ["Change_Control_Maturity", "Risk_Management_Maturity"]
NOMINAL_COLS   = ["Methodology_Used", "Funding_Source", "Contract_Type",
                  "Project_Phase", "Team_Colocation"]

# Zuordnung der Merkmale zu den fünf Bewertungskriterien nach
# Karrenbauer & Breitner (2022); bestimmt zugleich die Gruppierung der Oberfläche.
KUB_GROUPS = {
    "Complexity": ["Complexity_Score", "Integration_Complexity", "Cross_Functional_Dependencies",
                   "External_Dependencies_Count", "Technology_Familiarity", "Tech_Environment_Stability",
                   "Technical_Debt_Level", "Requirement_Stability", "Change_Request_Frequency",
                   "Team_Size", "Stakeholder_Count", "Geographical_Distribution"],
    "Efficiency": ["Project_Budget_USD", "Budget_Utilization_Rate", "Estimated_Timeline_Months",
                   "Resource_Availability", "Resource_Contention_Level", "Current_Phase_Duration_Months",
                   "Communication_Frequency", "Documentation_Quality", "Org_Process_Maturity",
                   "Team_Experience_Level", "Project_Manager_Experience", "Past_Similar_Projects",
                   "Previous_Delivery_Success_Rate", "Methodology_Used", "Team_Colocation"],
    "Risk":       ["Historical_Risk_Incidents", "Risk_Management_Maturity", "Change_Control_Maturity",
                   "Vendor_Reliability_Score", "Team_Turnover_Rate", "Market_Volatility",
                   "Industry_Volatility", "Regulatory_Compliance_Level", "Data_Security_Requirements",
                   "Seasonal_Risk_Factor"],
    "Strategy":   ["Executive_Sponsorship", "Stakeholder_Engagement_Level", "Key_Stakeholder_Availability",
                   "Funding_Source", "Contract_Type", "Client_Experience_Level",
                   "Organizational_Change_Frequency"],
    "Urgency":    ["Priority_Level", "Schedule_Pressure", "Project_Phase", "Project_Start_Month"],
}
ALL_FEATURES = [f for group in KUB_GROUPS.values() for f in group]

# Rangwerte der Risikostufen für den Erwartungswert-Score
CLASS_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Lesbare Beschriftungen, wo der Spaltenname allein nicht selbsterklärend ist
DISPLAY_LABEL = {
    "Project_Budget_USD": "Project Budget", "Budget_Utilization_Rate": "Budget Utilization",
    "Estimated_Timeline_Months": "Estimated Timeline (months)",
    "Current_Phase_Duration_Months": "Current Phase Duration (months)",
    "Previous_Delivery_Success_Rate": "Past Delivery Success Rate",
    "Org_Process_Maturity": "Org. Process Maturity",
    "Organizational_Change_Frequency": "Org. Change Frequency",
    "Project_Start_Month": "Project Start Month (1-12)",
    "Historical_Risk_Incidents": "Past Risk Incidents",
    "Cross_Functional_Dependencies": "Cross-Functional Dependencies",
}
SEMANTIC_TIP = {
    "Project_Start_Month": "Kalendermonat des Projektstarts (1 = Januar bis 12 = Dezember); bildet saisonale Effekte ab.",
    "Seasonal_Risk_Factor": "Saisonaler Risikofaktor des Projekts (etwa 1.0 bis 1.1).",
    "Budget_Utilization_Rate": "Anteil des Budgets, der voraussichtlich verbraucht wird (kann über 100 % liegen).",
    "Resource_Availability": "Verfügbarkeit benötigter Ressourcen (0 bis 100 %).",
    "Technical_Debt_Level": "Grad der angesammelten technischen Schulden (0 bis 100 %).",
    "Team_Turnover_Rate": "Erwartete Personalfluktuation im Team (0 bis 100 %).",
    "Vendor_Reliability_Score": "Zuverlässigkeit externer Dienstleister (0 bis 100 %).",
    "Communication_Frequency": "Kommunikationsfrequenz im Projekt (relativer Wert).",
    "Change_Request_Frequency": "Häufigkeit von Änderungsanfragen (relativer Wert).",
}


def label_of(f):
    return DISPLAY_LABEL.get(f, f.replace("_", " "))


def feat_type(f):
    if f in ORDINAL_ORDERS: return "ordinal"
    if f in MATURITY_COLS:  return "maturity"
    if f in NOMINAL_COLS:   return "nominal"
    return "numeric"


# ---------------------------------------------------------------------------
# Stufenbildung für numerische Merkmale
# ---------------------------------------------------------------------------
def _kind(series):
    """Bestimmt den Wertetyp, um Rundung und Formatierung passend zu wählen."""
    s = series.dropna()
    if (s == s.round()).all():                 return "int"
    if s.min() >= 0 and s.max() <= 1.3:        return "rate"
    if s.max() >= 1000:                        return "money"
    return "decimal"


def _round_nice(x, kind):
    """Rundet Stufengrenzen auf lesbare Werte."""
    if kind == "int":   return int(round(x))
    if kind == "rate":  return round(x * 20) / 20         # auf 5 Prozentpunkte
    if kind == "money": return round(x / 50000) * 50000   # auf 50.000
    return round(x * 2) / 2                               # auf 0.5


def _fmt(x, kind):
    if kind == "int":   return f"{int(x)}"
    if kind == "rate":  return f"{x*100:.0f}%"
    if kind == "money":
        return f"${x/1_000_000:.1f}M" if abs(x) >= 1_000_000 else f"${x/1000:.0f}k"
    return f"{x:.1f}"


def build_feature_spec(df):
    """Erzeugt je Merkmal die auswählbaren Stufen und deren Rückabbildung auf
    Modellwerte. Kategoriale Merkmale übernehmen ihre Kategorien direkt,
    numerische werden in gleich breite Intervalle über den beobachteten
    Wertebereich zerlegt. Gleich breite Intervalle statt Quantile sorgen dafür,
    dass die Abstände auf dem Regler den tatsächlichen Wertabständen entsprechen."""
    spec = {}
    for f in ALL_FEATURES:
        t = feat_type(f)
        if t in ("ordinal", "maturity"):
            opts = ORDINAL_ORDERS[f] if t == "ordinal" else MATURITY_ORDER
            spec[f] = {"type": t, "options": opts, "value_map": {o: o for o in opts}}
        elif t == "nominal":
            opts = sorted(df[f].dropna().unique().tolist())
            spec[f] = {"type": t, "options": opts, "value_map": {o: o for o in opts}}
        else:
            s = df[f].dropna()
            kind = _kind(s)
            uniq = np.sort(s.unique())
            options, vmap = [], {}
            if len(uniq) <= 6:
                # Wenige Ausprägungen: die echten Werte bilden die Stufen
                for u in uniq:
                    label = _fmt(u, kind)
                    if label in vmap:
                        continue
                    options.append(label)
                    vmap[label] = float(u)
            else:
                lo_r, hi_r = float(s.min()), float(s.max())
                n_bins = 4
                edges = np.linspace(lo_r, hi_r, n_bins + 1)
                if kind == "int":
                    edges = np.unique(np.round(edges))
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    last = (i == len(edges) - 2)
                    mask = (s >= lo) & (s <= hi) if last else (s >= lo) & (s < hi)
                    # Median des Intervalls als repräsentativer Wert für das Modell
                    rep = float(s[mask].median()) if mask.any() else float((lo + hi) / 2)
                    lo_n, hi_n = _round_nice(lo, kind), _round_nice(hi, kind)
                    label = (f"{_fmt(lo_n, kind)}\u2013{_fmt(hi_n, kind)}"
                             if _fmt(lo_n, kind) != _fmt(hi_n, kind) else _fmt(lo_n, kind))
                    if label in vmap:
                        continue
                    options.append(label)
                    vmap[label] = rep
            lo_disp = _fmt(_round_nice(float(s.min()), kind), kind)
            hi_disp = _fmt(_round_nice(float(s.max()), kind), kind)
            spec[f] = {"type": t, "options": options, "value_map": vmap, "kind": kind,
                       "min": float(s.min()), "max": float(s.max()),
                       "range_label": f"{lo_disp}\u2013{hi_disp}"}
    return spec


# ---------------------------------------------------------------------------
# Risikorichtung je Merkmal
# ---------------------------------------------------------------------------
def compute_directions(df, risk_rank):
    """Bestimmt über die Korrelation mit der Risikostufe, ob ein höherer
    Merkmalswert das Risiko erhöht (+1) oder senkt (-1). Nominale Merkmale
    besitzen keine Rangordnung und erhalten 0. Die Richtung steuert allein die
    Farbverläufe und Aggregatregler der Oberfläche, nicht die Vorhersage."""
    dirs = {}
    for f in ALL_FEATURES:
        t = feat_type(f)
        if t == "nominal":
            dirs[f] = 0
            continue
        if t == "ordinal":
            rank = df[f].map({v: i for i, v in enumerate(ORDINAL_ORDERS[f])})
        elif t == "maturity":
            rank = df[f].map({v: i for i, v in enumerate(MATURITY_ORDER)})
        else:
            rank = df[f]
        m = rank.notna() & risk_rank.notna()
        if m.sum() < 10:
            dirs[f] = 0
            continue
        c = np.corrcoef(rank[m].astype(float), risk_rank[m].astype(float))[0, 1]
        dirs[f] = 1 if c > 0.05 else (-1 if c < -0.05 else 0)
    return dirs


# ---------------------------------------------------------------------------
# Laden der Artefakte
# ---------------------------------------------------------------------------
def load_all(model_path="model_pipeline.joblib",
             meta_path="feature_defaults.joblib",
             csv_path="project_risk_raw_dataset.csv"):
    """Lädt Modell, Metadaten und Rohdaten. Die Rohdaten werden nur benötigt, um
    Stufengrenzen und Risikorichtungen abzuleiten, nicht für die Vorhersage."""
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    # keep_default_na=False: der Kategoriewert "None" muss erhalten bleiben
    raw = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)
    raw = raw[raw["Project_Type"] == "IT"]
    risk_rank = raw["Risk_Level"].map({"Low": 0, "Medium": 1, "High": 2, "Critical": 3})
    df = raw.drop(columns=["Project_ID", "Project_Type", "Risk_Level"])
    spec = build_feature_spec(df)
    directions = compute_directions(df, risk_rank)
    for f in ALL_FEATURES:
        spec[f]["direction"] = directions[f]
        spec[f]["label"] = label_of(f)
        tip = SEMANTIC_TIP.get(f, "")
        spec[f]["tooltip"] = (tip + " " if tip else "") + \
            "Stufen (links -> rechts): " + ", ".join(str(o) for o in spec[f]["options"])
    return model, meta, spec, df


# ---------------------------------------------------------------------------
# Vorhersage
# ---------------------------------------------------------------------------
def build_row(params, meta):
    """Erzeugt eine einzeilige Tabelle im Trainingsschema. Nicht gesetzte Merkmale
    werden mit dem Vorgabewert aus dem Training belegt. Spaltenauswahl und
    -reihenfolge stammen aus den Metadaten, damit sie exakt der Anpassung des
    Vorverarbeitungsschritts entsprechen."""
    cols = meta["all_features"]
    row = {f: (meta["defaults"][f] if params.get(f) is None else params[f]) for f in cols}
    df = pd.DataFrame([row], columns=cols)
    for c in list(ORDINAL_ORDERS) + MATURITY_COLS + NOMINAL_COLS:
        df[c] = df[c].astype(object)
    return df


def predict(model, meta, params):
    proba = model.predict_proba(build_row(params, meta))[0]
    return meta["target_order"], proba


def expected_score(order, proba):
    """Erwartungswert der Risikostufe (Low = 1 bis Critical = 4), gewichtet mit
    den Klassenwahrscheinlichkeiten."""
    return float(sum(CLASS_WEIGHT[c] * p for c, p in zip(order, proba)))


def level_from_score(score):
    if score < 1.5:  return "Low"
    if score < 2.5:  return "Medium"
    if score < 3.5:  return "High"
    return "Critical"


def elevated_prob(order, proba):
    """Wahrscheinlichkeit, dass ein Projekt in High oder Critical fällt."""
    d = dict(zip(order, proba))
    return d.get("High", 0) + d.get("Critical", 0)


def portfolio_metrics(per_project):
    """Aggregiert die Einzelvorhersagen auf Portfolioebene. Die Wahrscheinlichkeit
    für mindestens ein erhöhtes Projekt setzt Unabhängigkeit der Projekte voraus."""
    if not per_project:
        return None
    exp_scores = [expected_score(o, p) for o, p in per_project]
    port_score = float(np.mean(exp_scores))
    elevated = [elevated_prob(o, p) for o, p in per_project]
    return {
        "score": port_score,
        "level": level_from_score(port_score),
        "p_at_least_one_elevated": 1 - float(np.prod([1 - e for e in elevated])),
        "expected_elevated_count": float(np.sum(elevated)),
        "n": len(per_project),
    }


# ---------------------------------------------------------------------------
# SHAP-Erklärung
# ---------------------------------------------------------------------------
def _orig_feature(name):
    """Führt einen transformierten Spaltennamen auf das Originalmerkmal zurück,
    damit die One-Hot-Anteile eines Merkmals zusammengefasst werden."""
    core = name.split("__", 1)[1]
    if core in ORDINAL_ORDERS or core in MATURITY_COLS:
        return core
    for c in NOMINAL_COLS:
        if core.startswith(c + "_"):
            return c
    return core


def build_explainer(model, df, n_bg=100):
    """Erzeugt den LinearExplainer für die logistische Regression innerhalb der
    Pipeline. Als Hintergrundverteilung dient eine Stichprobe der Trainingsdaten."""
    import shap
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    bg = df.sample(min(n_bg, len(df)), random_state=42).copy()
    Xbg = prep.transform(bg)
    explainer = shap.LinearExplainer(clf, Xbg)
    return {"prep": prep, "explainer": explainer,
            "feat_names": list(prep.get_feature_names_out())}


def explain(ctx, meta, params, order, proba, top_n=None):
    """Ermittelt die Treiber des Gesamtrisikos, absteigend sortiert.

    Die klassenweisen SHAP-Werte werden mit dem zentrierten Rang der Risikostufe
    gewichtet und aufsummiert. Dadurch entsteht je Merkmal ein einzelner Wert:
    positiv bedeutet, das Merkmal verschiebt die Vorhersage zu höherem Risiko.
    Rückgabe: [(Merkmal, Beitrag, wurde_gesetzt)]."""
    Xt = ctx["prep"].transform(build_row(params, meta))
    sv = np.array(ctx["explainer"].shap_values(Xt))
    ncls = len(order)
    if sv.ndim == 3:
        mat = sv[:, 0, :] if sv.shape[0] == ncls else sv[0].T   # -> (Klassen, Merkmale)
    else:
        mat = sv[0][None, :]
    ranks = np.array([{"Low": 0, "Medium": 1, "High": 2, "Critical": 3}[c] for c in order], dtype=float)
    w = ranks - ranks.mean()
    risk_contrib = w @ mat
    agg = {}
    for val, name in zip(risk_contrib, ctx["feat_names"]):
        f = _orig_feature(name)
        agg[f] = agg.get(f, 0.0) + float(val)
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    out = [(f, v, params.get(f) is not None) for f, v in ranked]
    return out[:top_n] if top_n else out
