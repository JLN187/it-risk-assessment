"""Smoke-Test: laedt das exportierte Modell und macht eine Beispielvorhersage."""
import pandas as pd
import joblib

model = joblib.load("model_pipeline.joblib")
meta = joblib.load("feature_defaults.joblib")

# Rohdaten laden und exakt wie im Training vorbereiten
df = pd.read_csv("project_risk_raw_dataset.csv")
df = df[df["Project_Type"] == "IT"].drop(columns=["Project_ID", "Project_Type", "Risk_Level"])
for c in meta["maturity_cols"]:                       # Missing-Indikatoren ergaenzen
    df[c + "_missing"] = df[c].isna().astype(int)

row = df.iloc[[0]]                                     # ein Beispielprojekt
proba = model.predict_proba(row)[0]
order = meta["target_order"]
pred = order[proba.argmax()]

print("Vorhergesagte Risikoklasse:", pred)
for cls, p in zip(order, proba):
    print(f"  {cls:<10} {p:5.1%}{'   <== Maximum' if cls == pred else ''}")
