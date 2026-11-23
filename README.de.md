# Vorhersage von IT-Projektrisikoniveaus mittels maschinellen Lernens

[English](README.md) | **Deutsch**

**Live-Demo:** [it-risk-assessment.streamlit.app](https://it-risk-assessment.streamlit.app/)

Code zu meiner Bachelorarbeit an der Leibniz Universität Hannover, Institut für Wirtschaftsinformatik (IWI), 2026.

Das Projekt stuft IT-Projekte in vier Risikostufen ein (Low, Medium, High, Critical) und bindet die Vorhersage in das IT-Projektportfoliomanagement ein. Methodisch folgt die Arbeit dem Design Science Research nach Gregor & Hevner (2013). Alle Eingangsmerkmale sind den fünf Bewertungskriterien des wertorientierten Portfolioansatzes von Karrenbauer & Breitner (2022) zugeordnet: Komplexität, Effizienz, Risiko, Strategie und Dringlichkeit.

## Überblick

- Vergleich von logistischer Regression, Random Forest und XGBoost per verschachtelter fünffacher Kreuzvalidierung auf 1.381 IT-Projekten mit 48 Eingangsmerkmalen
- Bestes Modell (logistische Regression): Macro-F1 0,634 gegenüber 0,138 einer naiven Referenz, Macro-AUC 0,877
- 95,4 % der Fehleinstufungen liegen auf der benachbarten Risikostufe
- Erklärbare Vorhersagen mit SHAP (LinearExplainer für die logistische Regression, TreeExplainer für XGBoost)
- Streamlit-Prototyp zur Risikoanalyse einzelner Projekte und ganzer Portfolios, zweisprachig (DE/EN)
- Expertenbefragung (n = 15) und Praxisinterview zur Anwendbarkeit

## Ergebnisse

### Modellvergleich

| Modell | Macro-F1 | Balanced Accuracy | Macro-AUC |
|---|---|---|---|
| Logistische Regression | **0,634 ± 0,017** | **0,641** | **0,877** |
| XGBoost | 0,618 ± 0,016 | 0,608 | 0,855 |
| Random Forest | 0,536 ± 0,031 | 0,536 | 0,809 |
| Naive Referenz (immer Mehrheitsklasse) | 0,138 | | |

Mittelwerte über die äußeren Folds der verschachtelten Kreuzvalidierung; ± bezeichnet die Standardabweichung.

### Weitere Befunde

- **Fehlerprofil:** Accuracy 63,8 %. Von 500 Fehleinstufungen liegen 477 (95,4 %) eine Stufe neben der tatsächlichen. Genau ein Critical-Projekt wurde als Low eingestuft.
- **Lineare Datenstruktur:** Die logistische Regression übertrifft beide Ensemble-Verfahren. Eine Linearitätsprüfung (Interaktionsterme, Abgleich mit künstlichen linearen und nichtlinearen Referenzdaten) führt diesen Befund auf eine überwiegend lineare Struktur des Datensatzes zurück.
- **Stabilität der Erklärungen:** Die Rangfolgen der Merkmalswichtigkeit von logistischer Regression und XGBoost korrelieren mit Spearman ρ = 0,757.
- **Abgleich mit dem Bewertungsrahmen:** Der Score nach Karrenbauer & Breitner (2022) wurde auf 42 der 48 Merkmale übertragen. Er korreliert mit der tatsächlichen Risikostufe mit ρ = −0,301 (95-%-Bootstrap-Intervall −0,351 bis −0,253). Das negative Vorzeichen bedeutet, dass ein höherer Wertbeitrag mit geringerem Risiko einhergeht. Je Kriterium: Effizienz −0,485, Komplexität −0,395, Risiko −0,354, Strategie −0,276, Dringlichkeit +0,043 (nicht signifikant, p = 0,11).
- **Auswahlexperiment:** Werden Projekte absteigend nach Wertbeitrag ausgewählt, bis eine Budgetgrenze erreicht ist, sinkt der Anteil hoch und kritisch eingestufter Projekte von 39,9 % im Gesamtbestand auf 16,0 % bei 10 % des Budgets und auf 24,7 % bei 30 %.
- **Expertenbefragung:** 15 vollständige von 58 begonnenen Antworten. Das Vertrauen in die Modellausgabe wurde am niedrigsten bewertet. Für keines der 18 abgefragten Merkmale wurde die Datenverfügbarkeit hoch eingeschätzt (kein Mittelwert ab 3,50).

### Hypothesen

| | Hypothese | Ergebnis |
|---|---|---|
| H1 | Das ML-Modell übertrifft eine naive Referenz | Erfüllt |
| H2 | Ensemble-Verfahren übertreffen die logistische Regression | Widerlegt |
| H3 | Die nach SHAP relevanten Merkmale entsprechen den Kriterien nach Karrenbauer & Breitner | Differenziert gestützt: Auf Merkmalsebene sind alle fünf Kriterien vertreten, auf Kriterienebene zeigt die Dringlichkeit keinen Zusammenhang mit der Risikostufe |

## Prototyp

Die App läuft unter [it-risk-assessment.streamlit.app](https://it-risk-assessment.streamlit.app/) und lässt sich lokal mit `streamlit run app.py` starten.

- Portfolios anlegen und Projekte hinzufügen. Die Eingabe ist nach den fünf Kriterien gegliedert, wahlweise über einen Gesamtregler je Kriterium oder je Merkmal einzeln.
- Je Projekt: Wahrscheinlichkeiten der vier Risikostufen, Erwartungswert der Risikostufe, die stärksten risikoerhöhenden und risikosenkenden Merkmale nach SHAP sowie Hinweise zur Risikosenkung.
- Je Portfolio: Gesamtrisiko, Wahrscheinlichkeit für mindestens ein Projekt mit hohem oder kritischem Risiko (unter der Annahme unabhängiger Projekte), erwartete Anzahl solcher Projekte und Risikoverteilung.
- Optionale Restriktionen: Obergrenzen für Portfoliosummen (zum Beispiel Budget oder Teamgröße), Grenzwerte für Durchschnitte und eine Regel für regulatorische Anforderungen.
- Portfolios als JSON speichern und laden; Oberfläche auf Deutsch und Englisch.

Nicht gesetzte Merkmale erhalten den datensatztypischen Wert (Median bzw. häufigste Ausprägung). Die Farbverläufe und Gesamtregler beruhen auf einer bivariaten Korrelation mit der Risikostufe auf den Rohdaten (Schwelle |r| > 0,05), nicht auf dem Modell. Sie steuern nur die Oberfläche, nicht die Vorhersage.

## Aufbau des Repositorys

```
├── app.py                          Streamlit-Prototyp
├── model_bridge.py                 Verbindung zwischen Oberfläche und Modell (ohne Streamlit-Abhängigkeit)
├── masterskript.py                 Training, Evaluation, Abbildungen und Modellexport
├── test_model.py                   Funktionsprüfung des exportierten Modells
├── model_pipeline.joblib           Trainiertes Modell (Vorverarbeitung und logistische Regression)
├── feature_defaults.joblib         Vorgabewerte, Klassenreihenfolge und Merkmalsschema für den Prototyp
├── project_risk_raw_dataset.csv    Datensatz (4.000 Projekte, davon 1.381 IT-Projekte)
├── requirements.txt                Abhängigkeiten mit festen Versionen
├── LICENSE                         Lizenz des Codes (MIT)
├── LIZENZ_daten.md                 Lizenz des Datensatzes (CDLA-Permissive-1.0)
├── .streamlit/config.toml          Farbschema der Oberfläche
└── .github/workflows/keepalive.yml Hält die gehostete App erreichbar
```

## Installation und Ausführung

Voraussetzung: Python 3.11 oder neuer (getestet mit Python 3.13).

```bash
git clone https://github.com/JLN187/it-risk-assessment.git
cd it-risk-assessment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Trainiertes Modell und Datensatz liegen im Repository bei. Der Prototyp läuft deshalb ohne vorheriges Training.

1. **Prototyp starten:** `streamlit run app.py`
2. **Funktionsprüfung:** `python test_model.py`
3. **Ergebnisse reproduzieren (optional):** `python masterskript.py` führt alle Auswertungen erneut aus, überschreibt die beiden Modelldateien und legt `abbildungen/` (Konfusionsmatrix, Modellvergleich, SHAP-Beeswarm, Scoring-Abgleich und Auswahlexperiment als PDF) sowie `konsolenprotokoll.txt` an.

## Daten

Der Datensatz [„Project management Risk Raw“](https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw/data) stammt von ka66ledata und ist auf Kaggle unter der Lizenz CDLA-Permissive-1.0 veröffentlicht. Der vollständige Lizenztext liegt in `LIZENZ_daten.md`. Die Datei `project_risk_raw_dataset.csv` ist unverändert übernommen; die Auswahl der 1.381 IT-Projekte erfolgt erst im Code.

Der Datensatz enthält keine Fehlwerte. Der Wert „None“ in den Reifegradmerkmalen ist die unterste Stufe (kein Prozess vorhanden) und wird deshalb mit `keep_default_na=False` eingelesen.

## Methodik in Kürze

- Filterung auf IT-Projekte (1.381 von 4.000)
- Kodierung nach Skalenniveau: ordinale Merkmale behalten ihre Rangfolge, nominale werden per One-Hot kodiert, Standardisierung nur für die logistische Regression
- Vorverarbeitung vollständig innerhalb der Pipeline, damit keine Information aus den Testfolds in das Training gelangt
- Verschachtelte Kreuzvalidierung (5 äußere, 3 innere Folds) mit Klassengewichten gegen das Klassenungleichgewicht; SMOTE wurde geprüft, im finalen Modell aber nicht verwendet
- Gepaarter Signifikanztest der logistischen Regression gegen das beste Ensemble über eine 5 × 5 wiederholte Kreuzvalidierung
- Robustheitsprüfung der angenommenen Rangordnungen bei den erfahrungsbezogenen Merkmalen
- Einheitlich `random_state = 42` für die Reproduzierbarkeit

## Einschränkungen

- Synthetischer Datensatz (Kaggle): Die Ergebnisse zeigen die Methode, nicht die Güte auf realen Projektdaten. Die überwiegend lineare Struktur schränkt die Übertragbarkeit ein.
- Die Risikostufen sind im Datensatz vorgegeben; echte Erfolgs- oder Misserfolgsdaten liegen nicht vor.
- Validierung ausschließlich per Kreuzvalidierung, keine externe Validierung.
- Die Zuordnung der Merkmale zu den Kriterien und die Richtungen im Scoring sind interpretative Festlegungen.
- Der IT-Teilbestand enthält nur agile Arbeitsweisen (Agile, Scrum, Kanban). Ein Vergleich agiler mit klassischen Arbeitsweisen ist daher nicht möglich.

## Online-Anhang

[doi.org/10.17605/OSF.IO/ZWSFT](https://doi.org/10.17605/OSF.IO/ZWSFT): aggregierte Befragungsergebnisse, Dokumentation der KI-Nutzung, Suchprotokoll und Code

## Literatur

- Gregor, S., & Hevner, A. R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly, 37*(2), 337–355.
- Karrenbauer, C., & Breitner, M. H. (2022). Value-driven IT project portfolio management: Process model, evaluation framework, and decision support. *ICIS 2022 Proceedings*, 9. https://aisel.aisnet.org/icis2022/is_design/is_design/9

## Autor

Julien Aldag · Master Wirtschaftsingenieurwesen, Leibniz Universität Hannover · [LinkedIn](https://www.linkedin.com/in/julien-aldag)

Betreuung: M. Sc. Fenja Schulte · Prüfer: Prof. Dr. Michael H. Breitner

## Lizenz

- **Code:** MIT, siehe `LICENSE`
- **Datensatz:** CDLA-Permissive-1.0, siehe `LIZENZ_daten.md`; Urheber: ka66ledata ([Kaggle](https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw/data))
