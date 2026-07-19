"""
IT Portfolio Risk Assessment - Streamlit-Prototyp
Nutzt das trainierte Modell (model_pipeline.joblib) ueber model_bridge.py.
Aufbauend auf: Karrenbauer & Breitner (2022)
"""
import json
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
# Risikorichtung der KATEGORIE selbst: +1 = mehr davon -> mehr Risiko, -1 = mehr davon -> weniger Risiko
CAT_DIR = {"Complexity": +1, "Efficiency": -1, "Risk": +1, "Strategy": -1, "Urgency": +1}

# --------------------------------------------------------------------------------------
# i18n
# --------------------------------------------------------------------------------------
STR = {
 "en": {"app_title": "Portfolio Risk Analyzer", "configure": "Configure", "results": "Results",
   "portfolios": "Portfolios", "new_portfolio": "New Portfolio", "no_portfolios": "No portfolios yet.",
   "no_pf_sel": "No Portfolio Selected", "create_pf": "Create a new portfolio to get started",
   "create_btn": "Create New Portfolio", "pf_config": "Portfolio Configuration", "pf_name": "Portfolio Name",
   "portfolio_view": "Portfolio", "project_view": "New project", "no_projects_yet": "No projects added yet.",
   "project_default": "Project", "portfolio_default": "Portfolio", "choose": "Choose options",
   "more_hint": "The more features you set, the more reliable the prediction.",
   "warn_one": "Please set at least one feature.", "toast_added": "Project added",
   "not_calc": "No results yet. Configure a portfolio and press \u201cCalculate Results\u201d.",
   "tile_total": "Overall portfolio risk", "tile_pelev": "Chance of a high-risk project",
   "tile_exphigh": "Expected high-risk projects", "tile_projects": "Projects in portfolio",
   "tile_restr": "Restrictions violated",
   "added_projects": "Added Projects", "project_name": "Project Name",
   "min_hint": "Set at least {n} features. The more features you set, the more reliable the prediction.",
   "single": "Details", "fine_tune": "Set individually", "overall": "Overall",
   "apply": "Apply", "cancel": "Cancel", "reset_cat": "Back to overall slider",
   "n_individual": "{n} of {t} features set individually", "add_project": "Add Project", "calc": "Calculate Results",
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
   "load_err": "Model artifacts could not be loaded. Please run masterskript_final.py locally first.",
   "language": "Language", "save_pf": "Save portfolios", "load_pf": "Load portfolios (JSON)", "load_err_pf": "Could not read file.",
   "nom_warn": "Not set by this slider (no ranking possible): {names}. Use \u201cSet individually\u201d if needed.",
   "restrictions": "Restrictions", "restr_on": "Apply restrictions",
   "restr_hint": "Optional. Define portfolio limits; violations are flagged in the results.",
   "restr_a_pick": "Which totals to limit", "restr_a": "Portfolio totals (max)", "restr_b": "Portfolio averages", "restr_c": "Per-project rule",
   "restr_c_rule": "Flag projects with regulatory compliance at High or Critical",
   "restr_c_tip": "Regulatory requirements = how strongly the project is subject to laws, standards or audits (e.g. GDPR, ISO, financial supervision). Projects rated High or Critical are flagged because they usually need extra review, documentation and lead time.",
   "distribution": "Risk distribution", "restr_check": "Restriction Check", "ok": "OK", "violated": "VIOLATED",
   "limit": "limit", "actual": "actual", "min_avg": "min avg", "max_avg": "max avg",
   "flagged": "flagged project(s)", "no_restr": "No restrictions active.",
   "restr_note": "Values not set by you use the dataset's typical value, consistent with the prediction."},
 "de": {"app_title": "Portfolio-Risikoanalyse", "configure": "Konfigurieren", "results": "Ergebnisse",
   "portfolios": "Portfolios", "new_portfolio": "Neues Portfolio", "no_portfolios": "Noch keine Portfolios.",
   "no_pf_sel": "Kein Portfolio ausgew\u00e4hlt", "create_pf": "Erstelle ein Portfolio, um zu starten",
   "create_btn": "Neues Portfolio erstellen", "pf_config": "Portfolio-Konfiguration", "pf_name": "Portfolioname",
   "portfolio_view": "Portfolio", "project_view": "Neues Projekt", "no_projects_yet": "Noch keine Projekte hinzugef\u00fcgt.",
   "project_default": "Projekt", "portfolio_default": "Portfolio", "choose": "Bitte ausw\u00e4hlen",
   "more_hint": "Je mehr Merkmale gesetzt sind, desto zuverl\u00e4ssiger die Vorhersage.",
   "warn_one": "Bitte mindestens ein Merkmal setzen.", "toast_added": "Projekt hinzugefügt",
   "not_calc": "Noch keine Ergebnisse. Portfolio konfigurieren und \u201eErgebnisse berechnen\u201c dr\u00fccken.",
   "tile_total": "Gesamtrisiko des Portfolios", "tile_pelev": "Wahrscheinlichkeit f\u00fcr ein Hochrisikoprojekt",
   "tile_exphigh": "Erwartete Hochrisikoprojekte", "tile_projects": "Projekte im Portfolio",
   "tile_restr": "Verletzte Restriktionen",
   "added_projects": "Hinzugef\u00fcgte Projekte", "project_name": "Projektname",
   "min_hint": "Mindestens {n} Merkmale angeben. Je mehr Merkmale gesetzt sind, desto zuverl\u00e4ssiger die Vorhersage.",
   "single": "Details", "fine_tune": "Einzeln einstellen", "overall": "Gesamt",
   "apply": "\u00dcbernehmen", "cancel": "Abbrechen", "reset_cat": "Zur\u00fcck zum Gesamt-Regler",
   "n_individual": "{n} von {t} Merkmalen einzeln gesetzt", "add_project": "Projekt hinzuf\u00fcgen", "calc": "Ergebnisse berechnen",
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
   "load_err": "Modell-Artefakte konnten nicht geladen werden. Bitte zuerst masterskript_final.py lokal ausf\u00fchren.",
   "language": "Sprache", "save_pf": "Portfolios speichern", "load_pf": "Portfolios laden (JSON)", "load_err_pf": "Datei konnte nicht gelesen werden.",
   "nom_warn": "Von diesem Regler nicht gesetzt (keine Rangordnung m\u00f6glich): {names}. Bei Bedarf \u201eEinzeln einstellen\u201c nutzen.",
   "restrictions": "Restriktionen", "restr_on": "Restriktionen anwenden",
   "restr_hint": "Optional. Portfolio-Grenzwerte festlegen; Verletzungen werden in den Ergebnissen markiert.",
   "restr_a_pick": "Welche Summen begrenzen", "restr_a": "Portfoliosummen (max)", "restr_b": "Portfoliodurchschnitte", "restr_c": "Einzelprojekt-Regel",
   "restr_c_rule": "Projekte mit regulatorischen Anforderungen auf Hoch/Kritisch markieren",
   "restr_c_tip": "Regulatorische Anforderungen = wie stark das Projekt Gesetzen, Normen oder Pr\u00fcfungen unterliegt (z. B. DSGVO, ISO, Finanzaufsicht). Projekte mit Stufe Hoch oder Kritisch werden markiert, weil sie meist zus\u00e4tzliche Pr\u00fcfungen, Dokumentation und Vorlaufzeit ben\u00f6tigen.",
   "distribution": "Risikoverteilung", "restr_check": "Restriktionspr\u00fcfung", "ok": "OK", "violated": "VERLETZT",
   "limit": "Grenzwert", "actual": "Ist", "min_avg": "Min-\u00d8", "max_avg": "Max-\u00d8",
   "flagged": "markierte(s) Projekt(e)", "no_restr": "Keine Restriktionen aktiv.",
   "restr_note": "Nicht gesetzte Werte verwenden den datensatztypischen Wert \u2014 konsistent zur Vorhersage."},
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


EXPL = {
    "Complexity_Score": ("Gesamtkomplexität des Projekts auf einer Skala von ~2 (sehr einfach) bis 10 (extrem komplex).", "Overall project complexity on a scale from ~2 (very simple) to 10 (extremely complex)."),
    "Integration_Complexity": ("Aufwand, das System mit bestehenden Systemen zu verbinden (1 = trivial, 10 = sehr aufwendig).", "Effort to integrate the system with existing ones (1 = trivial, 10 = very demanding)."),
    "Cross_Functional_Dependencies": ("Anzahl abteilungsübergreifender Abhängigkeiten. Mehr = koordinationsintensiver.", "Number of cross-departmental dependencies. More = harder to coordinate."),
    "External_Dependencies_Count": ("Anzahl Abhängigkeiten von externen Parteien (Lieferanten, Partner).", "Number of dependencies on external parties (vendors, partners)."),
    "Technology_Familiarity": ("Wie vertraut das Team mit der eingesetzten Technologie ist (Neu bis Experte).", "How familiar the team is with the technology used (New to Expert)."),
    "Tech_Environment_Stability": ("Zustand der technischen Umgebung (Alt/Instabil bis Modern/Stabil).", "State of the technical environment (Legacy/Unstable to Modern/Stable)."),
    "Technical_Debt_Level": ("Angesammelte technische Schulden in Prozent (0 % = keine, 100 % = sehr hoch).", "Accumulated technical debt in percent (0% = none, 100% = very high)."),
    "Requirement_Stability": ("Wie stabil die Anforderungen sind (Volatil bis Stabil).", "How stable the requirements are (Volatile to Stable)."),
    "Change_Request_Frequency": ("Häufigkeit von Änderungsanfragen (relativer Wert; höher = mehr Änderungen).", "Frequency of change requests (relative value; higher = more changes)."),
    "Team_Size": ("Anzahl der Teammitglieder.", "Number of team members."),
    "Stakeholder_Count": ("Anzahl beteiligter Stakeholder. Mehr = abstimmungsintensiver.", "Number of stakeholders involved. More = more coordination."),
    "Geographical_Distribution": ("Grad der räumlichen Verteilung des Teams (mehr Standorte = höher).", "Degree of geographical distribution of the team (more sites = higher)."),
    "Project_Budget_USD": ("Gesamtbudget des Projekts.", "Total project budget."),
    "Budget_Utilization_Rate": ("Anteil des Budgets, der voraussichtlich verbraucht wird (kann über 100 % liegen).", "Share of budget expected to be consumed (can exceed 100%)."),
    "Estimated_Timeline_Months": ("Geplante Projektdauer in Monaten.", "Planned project duration in months."),
    "Resource_Availability": ("Verfügbarkeit der benötigten Ressourcen (0 % = keine, 100 % = voll).", "Availability of required resources (0% = none, 100% = full)."),
    "Resource_Contention_Level": ("Konkurrenz um Ressourcen mit anderen Projekten (Niedrig bis Hoch).", "Competition for resources with other projects (Low to High)."),
    "Current_Phase_Duration_Months": ("Dauer der aktuellen Projektphase in Monaten.", "Duration of the current project phase in months."),
    "Communication_Frequency": ("Kommunikationsfrequenz im Projekt (relativer Wert; höher = intensiver).", "Communication frequency in the project (relative value; higher = more intense)."),
    "Documentation_Quality": ("Qualität der Projektdokumentation (Schlecht bis Ausgezeichnet).", "Quality of project documentation (Poor to Excellent)."),
    "Org_Process_Maturity": ("Reifegrad der Organisationsprozesse (Ad-hoc bis Optimierend).", "Maturity of organisational processes (Ad-hoc to Optimising)."),
    "Team_Experience_Level": ("Erfahrungsniveau des Teams (Junior bis Experte).", "Experience level of the team (Junior to Expert)."),
    "Project_Manager_Experience": ("Erfahrung der Projektleitung (Junior-PM bis zertifiziertes PM).", "Experience of the project manager (Junior PM to Certified PM)."),
    "Past_Similar_Projects": ("Anzahl früherer ähnlicher Projekte (mehr = mehr Erfahrung).", "Number of past similar projects (more = more experience)."),
    "Previous_Delivery_Success_Rate": ("Erfolgsquote bisheriger Projektabschlüsse (0 % bis 100 %).", "Success rate of past project deliveries (0% to 100%)."),
    "Methodology_Used": ("Eingesetztes Vorgehensmodell (Agile, Scrum, Kanban).", "Project methodology used (Agile, Scrum, Kanban)."),
    "Team_Colocation": ("Räumliche Verteilung des Teams (voll vor Ort bis voll remote).", "Physical distribution of the team (fully colocated to fully remote)."),
    "Historical_Risk_Incidents": ("Anzahl früherer Risikovorfälle in vergleichbaren Projekten.", "Number of past risk incidents in comparable projects."),
    "Risk_Management_Maturity": ("Reife des Risikomanagements ('None' = kein Prozess vorhanden).", "Maturity of risk management ('None' = no process in place)."),
    "Change_Control_Maturity": ("Reife der Änderungssteuerung ('None' = kein Prozess vorhanden).", "Maturity of change control ('None' = no process in place)."),
    "Vendor_Reliability_Score": ("Zuverlässigkeit externer Dienstleister (0 % bis 100 %).", "Reliability of external vendors (0% to 100%)."),
    "Team_Turnover_Rate": ("Erwartete Personalfluktuation im Team (0 % bis 100 %).", "Expected team turnover (0% to 100%)."),
    "Market_Volatility": ("Volatilität des relevanten Marktes (0 % = stabil, 100 % = sehr volatil).", "Volatility of the relevant market (0% = stable, 100% = very volatile)."),
    "Industry_Volatility": ("Volatilität der Branche (Stabil bis Extrem).", "Volatility of the industry (Stable to Extreme)."),
    "Regulatory_Compliance_Level": ("Höhe der regulatorischen Anforderungen (Niedrig bis Kritisch).", "Level of regulatory requirements (Low to Critical)."),
    "Data_Security_Requirements": ("Höhe der Datensicherheitsanforderungen (Niedrig bis Streng).", "Level of data security requirements (Low to Strict)."),
    "Seasonal_Risk_Factor": ("Saisonaler Risikofaktor (100 % = neutral, 110 % = erhöht).", "Seasonal risk factor (100% = neutral, 110% = elevated)."),
    "Executive_Sponsorship": ("Rückhalt durch das Management (Schwach bis Stark).", "Backing by executive management (Weak to Strong)."),
    "Stakeholder_Engagement_Level": ("Einbindung der Stakeholder (Schlecht bis Ausgezeichnet).", "Level of stakeholder engagement (Poor to Excellent)."),
    "Key_Stakeholder_Availability": ("Verfügbarkeit zentraler Stakeholder (Schlecht bis Ausgezeichnet).", "Availability of key stakeholders (Poor to Excellent)."),
    "Funding_Source": ("Herkunft der Finanzierung (intern, extern, staatlich, gemischt).", "Source of funding (internal, external, government, mixed)."),
    "Contract_Type": ("Art des Vertrags (Festpreis, Zeit & Material, Cost-Plus, Hybrid).", "Type of contract (Fixed-Price, Time & Materials, Cost-Plus, Hybrid)."),
    "Client_Experience_Level": ("Erfahrung des Kunden mit solchen Projekten (Erstmalig bis Strategisch).", "Client's experience with such projects (First-time to Strategic)."),
    "Organizational_Change_Frequency": ("Häufigkeit organisatorischer Änderungen im Umfeld.", "Frequency of organisational change in the environment."),
    "Priority_Level": ("Priorität des Projekts (Niedrig bis Kritisch).", "Priority of the project (Low to Critical)."),
    "Schedule_Pressure": ("Terminlicher Druck (0 % = entspannt, höher = mehr Druck).", "Schedule pressure (0% = relaxed, higher = more pressure)."),
    "Project_Phase": ("Aktuelle Phase im Projektlebenszyklus (Initiierung bis Abschluss).", "Current phase in the project lifecycle (Initiation to Closure)."),
    "Project_Start_Month": ("Kalendermonat des Projektstarts (1 = Januar bis 12 = Dezember).", "Calendar month of project start (1 = January to 12 = December)."),
}

@st.cache_resource
def _load():
    model, meta, spec, df = mb.load_all()
    return model, meta, spec, mb.build_explainer(model, df)


@st.cache_data(show_spinner=False)
def cached_predict(params_items):
    """Vorhersage cachen: identische Projekt-Eingaben werden nicht neu berechnet
    (verhindert Neuaufbau aller Dashboard-Kacheln beim Tab-Wechsel)."""
    params = dict(params_items)
    order, proba = mb.predict(MODEL, META, params)
    return order, tuple(float(x) for x in proba)


def predict_proj(params):
    key = tuple(sorted((k, v) for k, v in params.items() if v is not None))
    order, proba = cached_predict(key)
    import numpy as _np
    return order, _np.array(proba)

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
 .block-container {{ padding-top:4rem; padding-bottom:4rem; max-width:1800px; }}
 h1,h2,h3 {{ font-weight:600; letter-spacing:-0.01em; }}
 .subtle {{ color:{MUTED}; font-size:0.95rem; }}
 .cat-header {{ color:{TEXT}; text-transform:uppercase; letter-spacing:0.14em; font-size:0.92rem; font-weight:800;
                 border-left:3px solid {HEAD}; padding-left:0.55rem; display:flex; align-items:center;
                 min-height:2.4rem; line-height:1; margin:0; }}
 .param-label {{ font-weight:600; font-size:0.88rem; margin:0.5rem 0 0.1rem 0; color:{TEXT}; }}
 .param-label span, .info span {{ cursor:help; }}
 .tick-wrap {{ position:relative; height:28px; margin:-0.5rem 9px 0.35rem 9px; }}
 .tick {{ position:absolute; transform:translateX(-50%); display:flex; flex-direction:column;
          align-items:center; font-size:0.66rem; color:{MUTED}; white-space:nowrap; }}
 .tick:first-child {{ transform:translateX(0); align-items:flex-start; }}
 .tick:last-child {{ transform:translateX(-100%); align-items:flex-end; }}
 .tick-hl {{ color:{TEXT} !important; font-weight:700; }}
 .tick-hl i {{ background:{HEAD} !important; width:2px !important; height:8px !important; }}
 .tick i {{ display:block; width:1px; height:5px; background:{BORDER}; margin-bottom:3px; }}
 .divider {{ height:1px; background:{BORDER}; margin:1rem 0; border:none; }}
 /* klar sichtbare Karten-Rahmen + verschachtelte Tiefe */
 [data-testid="stVerticalBlockBorderWrapper"] {{ border:1px solid {BORDER} !important; border-radius:10px !important; background:{PANEL}; }}
 /* Karten in einer Zeile gleich hoch */
 [data-testid="stHorizontalBlock"] {{ align-items:stretch; }}
 [data-testid="stHorizontalBlock"] > div,
 [data-testid="stColumn"] {{ display:flex; flex-direction:column; }}
 [data-testid="stHorizontalBlock"] > div > div,
 [data-testid="stColumn"] > div {{ flex:1; display:flex; flex-direction:column; }}
 [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {{ flex:1; height:auto; }}
 .stButton > button {{ font-size:1.02rem; }}
 [data-testid="stHorizontalBlock"] .stButton > button {{ font-weight:700; }}
 [data-testid="stNumberInput"] input {{ background:{PANEL_2} !important; color:{TEXT} !important; }}
 [data-baseweb="input"] {{ background:{PANEL_2} !important; }}
 [data-testid="stNumberInputContainer"] {{ background:{PANEL_2} !important; }}
 .nom-warn {{ color:#d9a441; font-size:0.74rem; line-height:1.2; margin:0.1rem 0 0.4rem 0;
             min-height:2.4rem; overflow:hidden; }}
 .icon-btn > button {{ font-size:1.15rem !important; font-weight:800 !important; padding:0.2rem 0 !important; }}
 .restr-label {{ font-size:0.78rem; color:{TEXT}; min-height:2.7rem; display:flex; align-items:flex-end;
                 line-height:1.15; margin-bottom:0.2rem; }}
 /* Multiselect-Tags (Restriktionen) lesbar: dunkler Chip, heller Text */
 [data-baseweb="tag"] {{ background-color:{PANEL_2} !important; color:{TEXT} !important;
                         border:1px solid {BORDER} !important; }}
 [data-baseweb="tag"] span {{ color:{TEXT} !important; }}
 [data-baseweb="tag"] svg {{ fill:{TEXT} !important; }}
 /* dezenter Details-Button */
 .subtle-btn > button {{ background:transparent !important; border:1px solid {BORDER} !important;
                         color:{MUTED} !important; font-size:0.82rem !important; padding:0.2rem 0.6rem !important; }}
 .subtle-btn > button:hover {{ border-color:{HEAD} !important; color:{TEXT} !important; }}
 .subtle-btn > button p {{ color:{MUTED} !important; }}
 /* Kategorie-Koerper (Aggregat vs. einzeln gesetzt) auf gleiche Mindesthoehe */
 .cat-body {{ min-height:6.2rem; }}
 .eqcard {{ min-height:12rem; }}
 @keyframes flashrow {{ 0% {{ background:rgba(91,181,107,0.55); }} 100% {{ background:transparent; }} }}
 .proj-row {{ border-radius:6px; padding:0.15rem 0; }}
 .proj-row.flash {{ animation:flashrow 1.6s ease-out 1; }}
 /* Datei-Upload zentriert */
 [data-testid="stFileUploaderDropzone"] {{ justify-content:center; text-align:center; }}
 [data-testid="stFileUploaderDropzone"] > div {{ display:flex; flex-direction:column; align-items:center; }}
 [data-testid="stFileUploaderDropzoneInstructions"] {{ align-items:center; text-align:center; }}
 [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {{ background:{BG}; }}
 /* aktiver Nav-/Sprach-Button: dunkles Highlight mit HELLEM Text (klarer Kontrast) */
 .stButton > button[kind="primary"] {{ background:{PANEL_2} !important; color:{TEXT} !important; border:1px solid {HEAD} !important; }}
 .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] div {{ color:{TEXT} !important; }}
 div[data-baseweb="slider"] div[role="slider"] {{ background:{HEAD} !important; border:2px solid {TEXT} !important; box-shadow:0 0 0 4px rgba(216,213,205,0.20) !important; }}
 [data-testid="stThumbValue"] {{ color:{TEXT} !important; font-weight:700 !important;
     background:{PANEL_2} !important; padding:0 5px; border-radius:4px; }}
 /* Streamlits eigene Min/Max-Beschriftung ausblenden - wir zeichnen eigene Ticks */
 [data-testid="stSliderTickBar"], [data-testid="stTickBar"] {{ display:none !important; }}
 [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ display:none !important; }}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("lang", "en")
ss.setdefault("portfolios", {})
ss.setdefault("active", None)
ss.setdefault("view", "Configure")
ss.setdefault("draft_id", 0)
ss.setdefault("pf_counter", 0)
MIN_FEATURES = 1


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


def default_proj_name(pf):
    return f"{T('project_default')} {len(pf['projects']) + 1}"


def new_portfolio():
    ss.pf_counter += 1
    pid = f"pf{ss.pf_counter}"
    ss.portfolios[pid] = {"name": f"Portfolio {ss.pf_counter}", "projects": []}
    ss.active, ss.view = pid, "Configure"


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<div style='font-size:1.35rem; font-weight:700; color:{TEXT}; "
                f"margin:0.2rem 0 1.1rem 0; text-align:center;'>{T('app_title')}</div>",
                unsafe_allow_html=True)
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

    # Abstand -> weniger wichtige Dinge nach unten
    st.markdown("<div style='height:38vh;'></div>", unsafe_allow_html=True)

    # Sprache knapp ueber dem Upload
    st.markdown(f"<div class='subtle' style='font-size:0.78rem; margin-bottom:0.3rem;'>{T('language')}</div>",
                unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    if lc1.button("\U0001F1EC\U0001F1E7 EN", key="lang_en", use_container_width=True,
                  type="primary" if ss.lang == "en" else "secondary"):
        ss.lang = "en"; st.rerun()
    if lc2.button("\U0001F1E9\U0001F1EA DE", key="lang_de", use_container_width=True,
                  type="primary" if ss.lang == "de" else "secondary"):
        ss.lang = "de"; st.rerun()

    if ss.portfolios:
        st.download_button(T("save_pf"),
                           data=json.dumps({"portfolios": ss.portfolios}, indent=2, ensure_ascii=False),
                           file_name="portfolios.json", mime="application/json",
                           use_container_width=True)
    up = st.file_uploader(T("load_pf"), type="json", key="pf_upload", label_visibility="collapsed")
    if up is not None and not ss.get("_loaded_once"):
        try:
            data = json.load(up)
            loaded = data.get("portfolios", {})
            if loaded:
                ss.portfolios.update(loaded)
                ss.active = list(loaded)[0]
                ss.pf_counter = max(ss.pf_counter, len(ss.portfolios))
                ss["_loaded_once"] = True
                st.rerun()
        except Exception:
            st.warning(T("load_err_pf"))



# --------------------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------------------
def _ticks(opts, highlight=None):
    n = len(opts)
    items = ""
    for i, o in enumerate(opts):
        left = 0 if n == 1 else i / (n - 1) * 100
        cls = "tick tick-hl" if i == highlight else "tick"
        items += f"<span class='{cls}' style='left:{left:.4f}%;'><i></i>{o}</span>"
    return f"<div class='tick-wrap'>{items}</div>"


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


def _mini_cat(crit):
    d = CAT_DIR.get(crit, 1)
    grad = f"to right,{GREEN},{RED}" if d > 0 else f"to right,{RED},{GREEN}"
    tip = (f"{T('overall')} {CAT(crit)}: "
           + (f"{vopt('Low')} \u2192 {T('dir_less')}, {vopt('Very High')} \u2192 {T('dir_more')}" if d > 0
              else f"{vopt('Low')} \u2192 {T('dir_more')}, {vopt('Very High')} \u2192 {T('dir_less')}"))
    return (f"<span title='{tip}' style='display:inline-block; width:26px; height:6px; "
            f"border-radius:3px; vertical-align:middle; margin-left:8px; "
            f"background:linear-gradient({grad});'></span>")


def _tip(feat):
    ex = EXPL.get(feat)
    lead = (ex[0] if ss.lang == "de" else ex[1]) + " \u2014 " if ex else ""
    return lead + T("levels") + ", ".join(vopt(o) for o in SPEC[feat]["options"])


def reliability(n):
    if n < 15:  return T("rel_low"), "#e5844d"
    if n < 30:  return T("rel_med"), "#d9a441"
    return T("rel_high"), GREEN


# --------------------------------------------------------------------------------------
# Restriktionen (angelehnt an Karrenbauer & Breitner 2022, Eq. 4 / Eq. 8)
# --------------------------------------------------------------------------------------
SUM_CANDIDATES = ["Team_Size", "Project_Budget_USD", "External_Dependencies_Count",
                  "Stakeholder_Count", "Cross_Functional_Dependencies"]                 # Typ A (addierbar)
SUM_DEFAULT = ["Team_Size", "Project_Budget_USD"]
AVG_FEATS = {"Resource_Availability": "min", "Budget_Utilization_Rate": "max",
             "Schedule_Pressure": "max"}                                            # Typ B
REG_FEAT  = "Regulatory_Compliance_Level"                                           # Typ C


def eff(params, feat):
    """Effektiver Wert = Nutzereingabe oder datensatztypischer Standardwert (wie beim Modell)."""
    v = params.get(feat)
    return META["defaults"][feat] if v is None else v


def _step(f):
    """Sinnvolle Schrittweite je Merkmal."""
    kind = SPEC[f].get("kind")
    if kind == "money":  return 50000.0
    if kind == "rate":   return 0.05
    if kind == "int":    return 1.0
    return 0.5


def _nice(v, kind):
    """Auf einen gut lesbaren Grenzwert runden."""
    if kind == "money":
        return round(v / 500_000) * 500_000 if v >= 500_000 else round(v / 100_000) * 100_000
    if kind == "rate":
        return round(v * 20) / 20
    if kind == "int":
        return float(round(v / 5) * 5) if v >= 20 else float(round(v))
    return round(v * 2) / 2


def default_limits(n, sum_feats):
    lim = {}
    for f in sum_feats:
        raw = float(META["defaults"][f]) * max(n, 1) * 1.25
        lim[f] = _nice(raw, SPEC[f].get("kind"))
    lim["Resource_Availability"] = 0.5
    lim["Budget_Utilization_Rate"] = 1.0
    lim["Schedule_Pressure"] = 0.2
    return lim


def render_restrictions(pf):
    r = pf.setdefault("restrictions", {"enabled": False, "limits": {}, "sum_feats": list(SUM_DEFAULT)})
    r.setdefault("sum_feats", list(SUM_DEFAULT))
    with st.container(border=True):
        h1, h2 = st.columns([0.42, 0.58])
        h1.markdown(f"<div class='cat-header'>{T('restrictions')}</div>", unsafe_allow_html=True)
        with h2:
            r["enabled"] = st.toggle(T("restr_on"), value=r["enabled"], key="restr_toggle")
        if not r["enabled"]:
            st.caption(T("restr_hint"))
            return
        r["sum_feats"] = st.multiselect(T("restr_a_pick"), SUM_CANDIDATES, default=r["sum_feats"],
                                        format_func=L, key="restr_sumpick", placeholder=T("choose"))
        base = default_limits(len(pf["projects"]), r["sum_feats"])
        for k, v in base.items():
            r["limits"].setdefault(k, v)
        if r["sum_feats"]:
            st.markdown(f"<div class='param-label'>{T('restr_a')}</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(r["sum_feats"]), 3))
            for i, f in enumerate(r["sum_feats"]):
                with cols[i % len(cols)]:
                    lo = 1.0 if f == "Team_Size" else 0.0     # >=1 Person bzw. >=0 Euro
                    unbounded = f in ("Team_Size", "Project_Budget_USD")
                    hi = None if unbounded else float(SPEC[f]["max"]) * max(len(pf["projects"]), 1)
                    cur = float(r["limits"].get(f, base.get(f, lo)))
                    cur = max(cur, lo) if hi is None else min(max(cur, lo), hi)
                    st.markdown(f"<div class='restr-label'>{L(f)}</div>", unsafe_allow_html=True)
                    r["limits"][f] = st.number_input(" ", min_value=lo, max_value=hi, value=cur,
                                                     step=_step(f), key=f"lim_{f}",
                                                     label_visibility="collapsed")
        st.markdown(f"<div class='param-label'>{T('restr_b')}</div>", unsafe_allow_html=True)
        cols = st.columns(len(AVG_FEATS))
        for c, (f, mode) in zip(cols, AVG_FEATS.items()):
            with c:
                tag = T("min_avg") if mode == "min" else T("max_avg")
                lo, hi = float(SPEC[f]["min"]), float(SPEC[f]["max"])
                cur = min(max(float(r["limits"][f]), lo), hi)
                is_rate = SPEC[f].get("kind") == "rate"
                st.markdown(f"<div class='restr-label'>{L(f)} ({tag}{', %' if is_rate else ''})</div>",
                            unsafe_allow_html=True)
                if is_rate:                                     # in Prozent anzeigen (wie die Slider)
                    val = st.number_input(" ", min_value=lo * 100, max_value=hi * 100,
                                          value=round(cur * 100, 1), step=5.0, key=f"lim_{f}",
                                          label_visibility="collapsed")
                    r["limits"][f] = val / 100
                else:
                    r["limits"][f] = st.number_input(" ", min_value=lo, max_value=hi, value=cur,
                                                     step=_step(f), key=f"lim_{f}",
                                                     label_visibility="collapsed")
        st.markdown(f"<div class='param-label'><span title='{T('restr_c_tip')}'>{T('restr_c')} &#9432;</span></div>",
                    unsafe_allow_html=True)
        st.caption(T("restr_c_rule"))


def check_restrictions(pf):
    """-> (rows, n_violations). rows: (label, actual_str, limit_str, ok)"""
    r = pf.get("restrictions", {})
    if not r.get("enabled"):
        return [], 0
    lim, rows, viol = r["limits"], [], 0
    projs = pf["projects"]
    for f in r.get("sum_feats", []):                       # Typ A: Summen
        total = sum(eff(p["params"], f) for p in projs)
        ok = total <= lim[f]
        viol += 0 if ok else 1
        rows.append((f"{L(f)}", f"{total:,.0f}", f"\u2264 {lim[f]:,.0f}", ok))
    for f, mode in AVG_FEATS.items():                     # Typ B: Durchschnitte
        avg = sum(eff(p["params"], f) for p in projs) / max(len(projs), 1)
        ok = (avg >= lim[f]) if mode == "min" else (avg <= lim[f])
        viol += 0 if ok else 1
        sign = "\u2265" if mode == "min" else "\u2264"
        if SPEC[f].get("kind") == "rate":
            rows.append((f"{L(f)}", f"{avg*100:.0f}%", f"{sign} {lim[f]*100:.0f}%", ok))
        else:
            rows.append((f"{L(f)}", f"{avg:.2f}", f"{sign} {lim[f]:.2f}", ok))
    flagged = [p["name"] for p in projs                   # Typ C: Einzelprojekt-Regulatorik
               if str(eff(p["params"], REG_FEAT)) in ("High", "Critical")]
    ok = len(flagged) == 0
    viol += 0 if ok else 1
    rows.append((f"<span title='{T('restr_c_tip')}'>{L(REG_FEAT)} &#9432;</span>",
                 f"{len(flagged)} {T('flagged')}", "0", ok))
    return rows, viol


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
        with c2:
            raw = st.text_input(" ", key=f"num_{pid}_{feat}",
                                placeholder=spec.get("range_label", ""),
                                label_visibility="collapsed", help=T("custom"))
        custom_val = None
        if raw:
            try:
                custom_val = max(spec["min"], min(spec["max"], float(raw.replace(",", "."))))
            except ValueError:
                custom_val = None
        with c1:
            if custom_val is not None:
                # Slider-Griff auf den zum Custom-Wert naechstgelegenen Checkpoint setzen
                reps = [spec["value_map"][o] for o in spec["options"]]
                nearest = min(range(len(reps)), key=lambda i: abs(reps[i] - custom_val))
                st.select_slider(" ", options=disp, value=disp[nearest + 1],
                                 key=f"insync_{pid}_{feat}", label_visibility="collapsed", disabled=True)
                st.markdown(_ticks(disp, highlight=nearest + 1), unsafe_allow_html=True)
            else:
                choice = st.select_slider(" ", options=disp, value="N/A",
                                          key=f"in_{pid}_{feat}", label_visibility="collapsed")
                st.markdown(_ticks(disp), unsafe_allow_html=True)
        if custom_val is not None:
            return custom_val
        return None if choice == "N/A" else spec["value_map"][back[choice]]
    choice = st.select_slider(" ", options=disp, value="N/A", key=f"in_{pid}_{feat}", label_visibility="collapsed")
    st.markdown(_ticks(disp), unsafe_allow_html=True)
    return None if choice == "N/A" else spec["value_map"][back[choice]]


def render_category_aggregate(pid, crit, feats):
    st.markdown(f"<div class='param-label'><span title='{T('tip_agg')}'>{T('overall')} {CAT(crit)} &#9432;</span>"
                f"{_mini_cat(crit)}</div>", unsafe_allow_html=True)
    disp = ["N/A"] + [vopt(x) for x in AGG_EN[1:]]
    choice = st.select_slider(" ", options=disp, value="N/A", key=f"agg_{pid}_{crit}", label_visibility="collapsed")
    st.markdown(_ticks(disp), unsafe_allow_html=True)
    idx = disp.index(choice)
    noms = [f for f in feats if SPEC[f]["type"] == "nominal"]
    # Platz fuer die Warnung immer freihalten -> Karten bleiben gleich hoch
    if idx > 0 and noms:
        st.markdown(f"<div class='nom-warn'>&#9888; "
                    f"{T('nom_warn', names=', '.join(L(f) for f in noms))}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nom-warn'>&nbsp;</div>", unsafe_allow_html=True)
    if idx == 0:
        return {f: None for f in feats}
    frac = (idx - 1) / (len(disp) - 2)              # 0 = wenig von der Kategorie ... 1 = viel
    # Kategoriestufe -> Risikoniveau (z. B. viel Effizienz = wenig Risiko)
    risk_frac = frac if CAT_DIR.get(crit, 1) > 0 else (1 - frac)
    out = {}
    for f in feats:
        if SPEC[f]["type"] == "nominal":
            out[f] = None          # keine Rangordnung -> kein passender Wert bestimmbar
            continue
        o = SPEC[f]["options"]
        # Risikoniveau -> Position auf der Merkmalsskala (Richtung des Merkmals beachten)
        ff = risk_frac if SPEC[f]["direction"] >= 0 else (1 - risk_frac)
        out[f] = SPEC[f]["value_map"][o[round(ff * (len(o) - 1))]]
    return out


HAS_DIALOG = hasattr(st, "dialog")


def open_cat_dialog(pid, crit, feats):
    """Popup mit allen Merkmalen der Kategorie (kein Springen der Seite)."""
    def _body():
        vals = {}
        cols = st.columns(2)
        for i, f in enumerate(feats):
            with cols[i % 2]:
                vals[f] = render_feature(pid, f)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button(T("apply"), key=f"ap_{pid}_{crit}", type="primary", use_container_width=True):
            ss.setdefault("catvals", {})[f"{pid}_{crit}"] = vals
            st.rerun()
        if c2.button(T("cancel"), key=f"cx_{pid}_{crit}", use_container_width=True):
            st.rerun()

    if HAS_DIALOG:
        @st.dialog(CAT(crit), width="large")
        def _dlg():
            _body()
        _dlg()
    else:                                    # Fallback fuer aeltere Streamlit-Versionen
        with st.expander(CAT(crit), expanded=True):
            _body()


def open_project_dialog(proj, order, proba):
    """Popup mit Treibern und gesetzten Merkmalen eines Projekts."""
    def _body():
        render_project_details(order, proba, proj["params"])
    if HAS_DIALOG:
        @st.dialog(proj["name"], width="large")
        def _dlg():
            _body()
        _dlg()
    else:
        with st.expander(proj["name"], expanded=True):
            _body()


def render_category_card(pid, crit, feats, draft):
    key = f"{pid}_{crit}"
    saved = ss.get("catvals", {}).get(key)
    with st.container(border=True):
        active = saved is not None and any(v is not None for v in saved.values())
        h1, h2 = st.columns([0.82, 0.18])
        badge = " \u2713" if active else ""
        h1.markdown(f"<div class='cat-header'>{CAT(crit)}"
                    f"<span style='color:{GREEN};'>{badge}</span></div>", unsafe_allow_html=True)
        with h2:
            if st.button("\u2699", key=f"btn_{key}", help=T("fine_tune"), use_container_width=True,
                         type="primary" if active else "secondary"):
                open_cat_dialog(pid, crit, feats)
        if saved is not None:
            n = sum(v is not None for v in saved.values())
            st.markdown("<div class='cat-body'>"
                        f"<div style='font-size:0.86rem; margin:0.4rem 0 0.9rem 0; color:{GREEN}; "
                        f"font-weight:600;'>{T('n_individual', n=n, t=len(feats))}</div>",
                        unsafe_allow_html=True)
            if st.button(T("reset_cat"), key=f"rst_{key}", use_container_width=True):
                ss["catvals"].pop(key, None); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            draft.update(saved)
        else:
            draft.update(render_category_aggregate(pid, crit, feats))


def render_configure(pf):
    if ss.pop("toast_add", False):
        st.toast(T("toast_added"), icon="\u2705")
    st.markdown(f"## {T('pf_config')}")
    left, right = st.columns([0.38, 0.62], gap="medium")

    # ---------------- Portfolio View (links) ----------------
    with left:
        with st.container(border=True):
            st.markdown(f"<div class='cat-header'>{T('portfolio_view')}</div>", unsafe_allow_html=True)
            _pfn = st.text_input(T("pf_name"), value="" if pf.get("auto_name") else pf["name"], placeholder=T("portfolio_default"))
            pf["auto_name"] = not _pfn.strip()
            pf["name"] = _pfn.strip() or T("portfolio_default")
            if pf["projects"]:
                st.markdown(f"<div class='param-label'>{T('added_projects')}</div>", unsafe_allow_html=True)
                flash_i = ss.pop("flash_idx", None)
                for i, proj in enumerate(pf["projects"]):
                    disp = f"{T('project_default')} {i + 1}" if proj.get("auto") else proj["name"]
                    flash = " flash" if i == flash_i else ""
                    c1, c2 = st.columns([0.8, 0.2], vertical_alignment="center")
                    c1.markdown(f"<div class='proj-row{flash}' style='color:{TEXT}; font-size:0.9rem; font-weight:700;'>"
                                f"<span style='border-left:3px solid {HEAD}; padding-left:0.5rem;'>"
                                f"{disp}</span></div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
                        if st.button("\U0001F5D1\uFE0F", key=f"del_{i}", use_container_width=True):
                            pf["projects"].pop(i); st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption(T("no_projects_yet"))
            render_restrictions(pf)          # Restriktionen gehoeren sichtbar zur Portfolio-Ebene

    # ---------------- Project View (rechts) ----------------
    pid = ss.draft_id
    with right:
        with st.container(border=True):
            st.markdown(f"<div class='cat-header'>{T('project_view')}</div>", unsafe_allow_html=True)
            proj_name = st.text_input(T("project_name"), value="",
                                      placeholder=default_proj_name(pf), key=f"pname_{pid}")
            st.caption(T("more_hint"))

            draft = {}
            crits = list(mb.KUB_GROUPS.items())
            for rs in range(0, len(crits), 2):                     # 2-Spalten-Raster
                cols = st.columns(2)
                for col, (crit, feats) in zip(cols, crits[rs:rs + 2]):
                    with col:
                        render_category_card(pid, crit, feats, draft)

        n_set = sum(v is not None for v in draft.values())
        _, a_col, b_col, _ = st.columns([0.12, 0.38, 0.38, 0.12])
        with a_col:
            if st.button(f"{T('add_project')} ({n_set})", use_container_width=True, type="primary"):
                if n_set < MIN_FEATURES:
                    st.warning(T("warn_one"))
                else:
                    name = proj_name.strip() or default_proj_name(pf)
                    pf["projects"].append({"name": name, "params": draft,
                                           "auto": not proj_name.strip()})
                    ss["flash_idx"] = len(pf["projects"]) - 1     # zuletzt hinzugefuegtes hervorheben
                    ss["toast_add"] = True                         # Toast nach dem Rerun zeigen
                    ss.draft_id += 1; st.rerun()
        with b_col:
            if st.button(T("calc"), use_container_width=True):
                if not pf["projects"]:
                    st.warning(T("warn_add"))
                else:
                    pf["calculated"] = True
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
    order, proba = predict_proj(proj["params"])
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
        with head_r:
            st.markdown("<div class='subtle-btn'>", unsafe_allow_html=True)
            if st.button(T("single"), key=f"pdet_{ss.active}_{i}", use_container_width=True):
                open_project_dialog(proj, order, proba)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(prob_bars(order, proba, pred) + "<div style='height:0.7rem;'></div>",
                    unsafe_allow_html=True)


def _tile(label, value, color, tip=""):
    t = f" title='{tip}'" if tip else ""
    info = " &#9432;" if tip else ""
    return (f"<div style='background:{PANEL}; border:1px solid {BORDER}; border-radius:10px;"
            f" padding:0.7rem 0.9rem; height:100%;'>"
            f"<div class='subtle'{t} style='font-size:0.78rem; cursor:{'help' if tip else 'default'};'>"
            f"{label}{info}</div>"
            f"<div style='font-size:1.6rem; font-weight:700; color:{color}; line-height:1.5;'>{value}</div></div>")


def _dist_chart(per_project):
    counts = {c: 0 for c in META["target_order"]}
    for order, proba in per_project:
        counts[order[list(proba).index(max(proba))]] += 1
    total = max(sum(counts.values()), 1)
    rows = ""
    for cls, n in counts.items():
        c = LEVEL_COLORS[cls]
        rows += (f"<div style='display:flex; align-items:center; gap:0.7rem; margin:0.3rem 0;'>"
                 f"<div style='width:74px; color:{MUTED}; font-size:0.8rem;'>{vopt(cls)}</div>"
                 f"<div style='flex:1; background:{PANEL_2}; border-radius:5px; height:12px;'>"
                 f"<div style='width:{n/total*100:.0f}%; background:{c}; height:12px; border-radius:5px;'></div></div>"
                 f"<div style='width:26px; text-align:right; color:{TEXT}; font-size:0.8rem; font-weight:600;'>{n}</div>"
                 f"</div>")
    return rows


def render_results(pf):
    st.markdown(f"## {T('risk_results')}")
    st.markdown(f"<div class='subtle'>{pf['name']}</div>", unsafe_allow_html=True)
    if not pf["projects"]:
        st.info(T("no_projects")); return
    if not pf.get("calculated"):
        st.info(T("not_calc")); return

    per_project = [predict_proj(p["params"]) for p in pf["projects"]]
    pm = mb.portfolio_metrics(per_project)
    p_color = LEVEL_COLORS[pm["level"]]
    rows, viol = check_restrictions(pf)

    # ---- Zeile 1: KPI-Kacheln ----
    k = st.columns(4)
    k[0].markdown(_tile(T("tile_total"), vopt(pm["level"]), p_color), unsafe_allow_html=True)
    k[1].markdown(_tile(T("tile_pelev"), f"{pm['p_at_least_one_elevated']:.0%}", TEXT, T("tip_elev")),
                  unsafe_allow_html=True)
    k[2].markdown(_tile(T("tile_exphigh"), f"{pm['expected_elevated_count']:.1f}", TEXT, T("tip_cnt")),
                  unsafe_allow_html=True)
    if rows:
        k[3].markdown(_tile(T("tile_restr"), f"{viol}" if viol else T("ok"),
                            RED if viol else GREEN), unsafe_allow_html=True)
    else:
        k[3].markdown(_tile(T("tile_restr"), "\u2014", MUTED, T("no_restr")), unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

    # ---- Zeile 2: Verteilung + Restriktionspruefung nebeneinander ----
    c1, c2 = st.columns([0.42, 0.58], gap="medium", vertical_alignment="top")
    with c1:
        with st.container(border=True):
            st.markdown(f"<div class='eqcard'><div class='cat-header'>{T('distribution')} \u00b7 {pm['n']}</div>"
                        f"{_dist_chart(per_project)}</div>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            body = f"<div class='cat-header'>{T('restr_check')}</div>"
            if not rows:
                body += f"<div class='subtle' style='margin-top:0.5rem;'>{T('no_restr')}</div>"
            else:
                for label, actual, limit, ok in rows:
                    c = GREEN if ok else RED
                    body += (f"<div style='display:flex; align-items:center; gap:0.6rem; padding:0.22rem 0;"
                             f" border-bottom:1px solid {BORDER};'>"
                             f"<div style='flex:1; color:{TEXT}; font-size:0.8rem;'>{label}</div>"
                             f"<div style='width:110px; text-align:right; color:{MUTED}; font-size:0.75rem;'>{actual}</div>"
                             f"<div style='width:100px; text-align:right; color:{MUTED}; font-size:0.75rem;'>{limit}</div>"
                             f"<div style='width:64px; text-align:right; color:{c}; font-weight:700; font-size:0.75rem;'>"
                             f"{T('ok') if ok else T('violated')}</div></div>")
                body += f"<div class='subtle' style='margin-top:0.4rem; font-size:0.72rem;'>{T('restr_note')}</div>"
            st.markdown(body, unsafe_allow_html=True)

    # ---- Zeile 3: Projektkarten (2 Spalten) ----
    st.markdown(f"### {T('breakdown')}")
    projs = list(enumerate(pf["projects"]))
    for rs in range(0, len(projs), 2):
        cols = st.columns(2)
        for col, (i, proj) in zip(cols, projs[rs:rs + 2]):
            with col:
                render_project_card(i, proj)


# ======================================================================================
# ROUTER (Nav erst nach Berechnung; davor nur Konfiguration)
# ======================================================================================
active_pf = ss.portfolios.get(ss.active) if ss.active in ss.portfolios else None
show_nav = bool(active_pf and active_pf.get("calculated"))

if show_nav:
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
else:
    ss.view = "Configure"       # ohne Ergebnisse kein Results-Tab
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

if ss.active is None or ss.active not in ss.portfolios:
    render_empty_state()
elif ss.view == "Configure":
    render_configure(ss.portfolios[ss.active])
else:
    render_results(ss.portfolios[ss.active])
