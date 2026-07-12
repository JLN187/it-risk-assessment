"""
model_bridge.py - Verbindung zwischen UI-Eingaben und dem trainierten Modell.
Enthaelt KEINEN Streamlit-Code, damit die Vorhersagelogik separat testbar ist.
"""
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Feature-Wissen (muss zum Masterskript passen)
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
MATURITY_ORDER = ["Basic", "Formal", "Advanced"]
MATURITY_COLS  = ["Change_Control_Maturity", "Risk_Management_Maturity"]
NOMINAL_COLS   = ["Methodology_Used", "Funding_Source", "Contract_Type",
                  "Project_Phase", "Team_Colocation"]

# Gruppierung nach den fuenf K&B-Kriterien (= UI-Struktur)
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

# Risikoklassen-Gewichte fuer Erwartungswert-Score
CLASS_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------
def load_all(model_path="model_pipeline.joblib",
             meta_path="feature_defaults.joblib",
             csv_path="project_risk_raw_dataset.csv"):
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    df = pd.read_csv(csv_path)
    df = df[df["Project_Type"] == "IT"].drop(columns=["Project_ID", "Project_Type", "Risk_Level"])
    spec = build_feature_spec(df)
    return model, meta, spec


def feat_type(f):
    if f in ORDINAL_ORDERS: return "ordinal"
    if f in MATURITY_COLS:  return "maturity"
    if f in NOMINAL_COLS:   return "nominal"
    return "numeric"


def _fmt(x):
    """Zahl kompakt formatieren."""
    ax = abs(x)
    if ax >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if ax >= 1_000:     return f"{x/1_000:.0f}k"
    if ax >= 10:        return f"{x:.0f}"
    if ax >= 1:         return f"{x:.1f}"
    return f"{x:.2f}"


def build_feature_spec(df):
    """Pro Feature: Typ, waehlbare Optionen (Labels) und Mapping Label->Modellwert."""
    spec = {}
    for f in ALL_FEATURES:
        t = feat_type(f)
        if t == "ordinal":
            opts = ORDINAL_ORDERS[f]
            spec[f] = {"type": t, "options": opts, "value_map": {o: o for o in opts}}
        elif t == "maturity":
            opts = MATURITY_ORDER
            spec[f] = {"type": t, "options": opts, "value_map": {o: o for o in opts}}
        elif t == "nominal":
            opts = sorted(df[f].dropna().unique().tolist())
            spec[f] = {"type": t, "options": opts, "value_map": {o: o for o in opts}}
        else:  # numeric -> 4 Quartilsbereiche mit repraesentativem Wert
            s = df[f].dropna()
            qs = np.quantile(s, [0, .25, .5, .75, 1.0])
            options, vmap = [], {}
            for i in range(4):
                lo, hi = qs[i], qs[i + 1]
                mask = (s >= lo) & (s <= hi) if i == 3 else (s >= lo) & (s < hi)
                rep = float(s[mask].median()) if mask.any() else float((lo + hi) / 2)
                label = f"{_fmt(lo)}\u2013{_fmt(hi)}"
                options.append(label)
                vmap[label] = rep
            spec[f] = {"type": t, "options": options, "value_map": vmap}
    return spec


# ---------------------------------------------------------------------------
# Vorhersage
# ---------------------------------------------------------------------------
def build_row(params, meta):
    """params: feature -> Modellwert oder None. Baut vollstaendige 48+2-Feature-Zeile."""
    row = {}
    for f in ALL_FEATURES:
        v = params.get(f)
        if v is None:
            row[f] = np.nan if f in MATURITY_COLS else meta["defaults"][f]
        else:
            row[f] = v
    for c in MATURITY_COLS:                     # Missing-Indikatoren
        row[c + "_missing"] = 1 if params.get(c) is None else 0
    df = pd.DataFrame([row])
    # kategoriale Spalten als object typisieren (sonst bricht der Encoder bei All-NaN-Spalten)
    for c in list(ORDINAL_ORDERS) + MATURITY_COLS + NOMINAL_COLS:
        df[c] = df[c].astype(object)
    return df


def predict(model, meta, params):
    """Gibt (Klassen-Reihenfolge, Wahrscheinlichkeiten) zurueck."""
    proba = model.predict_proba(build_row(params, meta))[0]
    return meta["target_order"], proba


def expected_score(order, proba):
    return float(sum(CLASS_WEIGHT[c] * p for c, p in zip(order, proba)))


def level_from_score(score):
    if score < 1.5:  return "Low"
    if score < 2.5:  return "Medium"
    if score < 3.5:  return "High"
    return "Critical"


def elevated_prob(order, proba):
    """P(erhoehtes Risiko) = P(High) + P(Critical) fuer ein Projekt."""
    d = dict(zip(order, proba))
    return d.get("High", 0) + d.get("Critical", 0)


def portfolio_metrics(per_project):
    """per_project: Liste von (order, proba). Aggregiert das Portfolio."""
    if not per_project:
        return None
    exp_scores = [expected_score(o, p) for o, p in per_project]
    port_score = float(np.mean(exp_scores))
    elevated = [elevated_prob(o, p) for o, p in per_project]
    p_at_least_one = 1 - float(np.prod([1 - e for e in elevated]))  # Baumdiagramm-Pfad
    exp_count_elevated = float(np.sum(elevated))
    return {
        "score": port_score,
        "level": level_from_score(port_score),
        "p_at_least_one_elevated": p_at_least_one,
        "expected_elevated_count": exp_count_elevated,
        "n": len(per_project),
    }
