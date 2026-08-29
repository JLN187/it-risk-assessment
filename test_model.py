"""
Funktionsprüfung: lädt das exportierte Modell und erzeugt eine Beispielvorhersage.

Entstanden im Rahmen der Bachelorarbeit "Vorhersage von
IT-Projektrisikoniveaus mittels maschinellen Lernens", Leibniz Universität
Hannover, Institut für Wirtschaftsinformatik, 2026
"""
import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_pipeline.joblib"))
meta = joblib.load(os.path.join(BASE_DIR, "feature_defaults.joblib"))

# keep_default_na=False: der Kategoriewert "None" bezeichnet die unterste
# Reifestufe und muss als Zeichenkette erhalten bleiben.
df = pd.read_csv(os.path.join(BASE_DIR, "project_risk_raw_dataset.csv"),
                 keep_default_na=False, na_filter=False)
df = df[df["Project_Type"] == "IT"]
df = df[meta["all_features"]]        # Spaltenauswahl und -reihenfolge wie im Training

row = df.iloc[[0]]
proba = model.predict_proba(row)[0]
order = meta["target_order"]
pred = order[proba.argmax()]

print("Vorhergesagte Risikoklasse:", pred)
for cls, p in zip(order, proba):
    print(f"  {cls:<10} {p:5.1%}{'   <== Maximum' if cls == pred else ''}")
