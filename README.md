# Predicting IT Project Risk Levels with Machine Learning

**English** | [Deutsch](README.de.md)

**Live demo:** [it-risk-assessment.streamlit.app](https://it-risk-assessment.streamlit.app/)

Code for my bachelor's thesis *"Vorhersage von IT-Projektrisikoniveaus mittels maschinellen Lernens"* (Predicting IT project risk levels with machine learning), Leibniz Universität Hannover, Institute of Information Systems (IWI), 2026.

The project classifies IT projects into four risk levels (Low, Medium, High, Critical) and integrates the prediction into IT project portfolio management. It follows the Design Science Research approach of Gregor & Hevner (2013). All input features are mapped to the five evaluation criteria of the value-driven portfolio framework by Karrenbauer & Breitner (2022): complexity, efficiency, risk, strategy and urgency.

## Highlights

- Comparison of logistic regression, random forest and XGBoost using nested 5-fold cross-validation on 1,381 IT projects with 48 input features
- Best model (logistic regression): Macro-F1 0.634 vs. 0.138 for a naive baseline, Macro-AUC 0.877
- 95.4 % of misclassifications fall on an adjacent risk level
- Explainable predictions with SHAP (LinearExplainer for logistic regression, TreeExplainer for XGBoost)
- Streamlit decision support prototype for single projects and whole portfolios, bilingual (EN/DE)
- Expert survey (n = 15) and a practitioner interview on practical applicability

## Results

### Model comparison

| Model | Macro-F1 | Balanced Accuracy | Macro-AUC |
|---|---|---|---|
| Logistic regression | **0.634 ± 0.017** | **0.641** | **0.877** |
| XGBoost | 0.618 ± 0.016 | 0.608 | 0.855 |
| Random forest | 0.536 ± 0.031 | 0.536 | 0.809 |
| Naive baseline (always majority class) | 0.138 | | |

Mean over the outer folds of the nested cross-validation; ± is the standard deviation.

### Further findings

- **Error profile:** Accuracy 63.8 %. Of 500 misclassifications, 477 (95.4 %) are one level away from the true level. Exactly one Critical project was classified as Low.
- **Linear data structure:** Logistic regression outperforms both tree ensembles. A linearity check (interaction terms, comparison with synthetic linear and non-linear reference data) attributes this to a predominantly linear structure of the dataset.
- **Stability of explanations:** The feature importance rankings of logistic regression and XGBoost correlate at Spearman ρ = 0.757.
- **Comparison with the evaluation framework:** The Karrenbauer & Breitner (2022) score was operationalised on 42 of the 48 features. It correlates with the true risk level at ρ = −0.301 (95 % bootstrap interval −0.351 to −0.253). The negative sign means that a higher value contribution goes along with lower risk. Per criterion: efficiency −0.485, complexity −0.395, risk −0.354, strategy −0.276, urgency +0.043 (not significant, p = 0.11).
- **Selection experiment:** When projects are selected in descending order of value contribution until a budget limit is reached, the share of High and Critical projects drops from 39.9 % in the full dataset to 16.0 % at 10 % of the total budget and to 24.7 % at 30 %.
- **Expert survey:** 15 complete responses out of 58 started. Trust in the model output received the lowest rating. Data availability was not rated high for any of the 18 features asked about (no mean of 3.50 or above).

### Hypotheses

| | Hypothesis | Outcome |
|---|---|---|
| H1 | The ML model outperforms a naive baseline | Supported |
| H2 | Ensemble methods outperform logistic regression | Rejected |
| H3 | The features relevant according to SHAP correspond to the Karrenbauer & Breitner criteria | Partially supported: all five criteria are represented at feature level, but at criterion level urgency shows no relationship with the risk level |

## Prototype

The app runs at [it-risk-assessment.streamlit.app](https://it-risk-assessment.streamlit.app/) and can be started locally with `streamlit run app.py`.

- Create portfolios and add projects. Input is grouped by the five criteria, either via one overall slider per criterion or feature by feature.
- Per project: probabilities of the four risk levels, expected risk score, the strongest risk-increasing and risk-reducing features according to SHAP, and hints on how to reduce risk.
- Per portfolio: overall risk level, probability of at least one High or Critical project (assuming independent projects), expected number of such projects and risk distribution.
- Optional restrictions: caps on portfolio totals (e.g. budget, team size), limits on averages and a rule for regulatory requirements.
- Save and load portfolios as JSON; interface in English and German.

Features left unset take the typical value of the dataset (median or most frequent category). The colour gradients and overall sliders are based on a bivariate correlation with the risk level on the raw data (threshold |r| > 0.05), not on the model. They only drive the interface, not the prediction.

## Repository structure

```
├── app.py                          Streamlit prototype
├── model_bridge.py                 Link between interface and model (no Streamlit dependency)
├── masterskript.py                 Training, evaluation, figures and model export
├── test_model.py                   Smoke test of the exported model
├── model_pipeline.joblib           Trained model (preprocessing and logistic regression)
├── feature_defaults.joblib         Default values, class order and feature schema for the prototype
├── project_risk_raw_dataset.csv    Dataset (4,000 projects, 1,381 of them IT projects)
├── requirements.txt                Pinned dependencies
├── LICENSE                         Code license (MIT)
├── LIZENZ_daten.md                 Dataset license (CDLA-Permissive-1.0)
├── .streamlit/config.toml          Colour theme of the interface
└── .github/workflows/keepalive.yml Keeps the hosted app awake
```

## Installation and usage

Requires Python 3.11 or newer (tested with Python 3.13).

```bash
git clone https://github.com/JLN187/it-risk-assessment.git
cd it-risk-assessment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The trained model and the dataset are included, so the prototype runs without training first.

1. **Start the prototype:** `streamlit run app.py`
2. **Smoke test:** `python test_model.py`
3. **Reproduce the results (optional):** `python masterskript.py` reruns all analyses, overwrites both model files and creates `abbildungen/` (confusion matrix, model comparison, SHAP beeswarm, scoring comparison and selection experiment as PDF) and `konsolenprotokoll.txt`.

## Data

The dataset ["Project management Risk Raw"](https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw/data) was created by ka66ledata and is published on Kaggle under the CDLA-Permissive-1.0 license. The full license text is in `LIZENZ_daten.md`. The file `project_risk_raw_dataset.csv` is included unchanged; the 1,381 IT projects are selected in the code.

The dataset contains no missing values. The value "None" in the maturity features is the lowest level (no process in place) and is therefore read with `keep_default_na=False`.

## Methodology in brief

- Filtering to IT projects (1,381 of 4,000)
- Encoding by scale level: ordinal features keep their rank order, nominal features are one-hot encoded, standardisation only for logistic regression
- All preprocessing inside the pipeline, so no information leaks from the test folds
- Nested cross-validation (5 outer, 3 inner folds) with class weights against class imbalance; SMOTE was tested but not used in the final model
- Paired significance test of logistic regression vs. the best ensemble over 5 × 5 repeated cross-validation
- Robustness check of the assumed rank orders for the experience-related features
- `random_state = 42` throughout for reproducibility

## Limitations

- Synthetic dataset (Kaggle): the results demonstrate the method, not the performance on real project data. The predominantly linear structure limits transferability.
- The risk levels are given in the dataset; there are no real success or failure labels.
- Validation by cross-validation only, no external validation.
- The mapping of features to criteria and the directions used in the scoring are interpretive decisions.
- The IT subset contains only agile ways of working (Agile, Scrum, Kanban), so a comparison of agile and traditional approaches is not possible.

## Online appendix

[doi.org/10.17605/OSF.IO/ZWSFT](https://doi.org/10.17605/OSF.IO/ZWSFT): aggregated survey results, documentation of AI use, search protocol and code

## References

- Gregor, S., & Hevner, A. R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly, 37*(2), 337–355.
- Karrenbauer, C., & Breitner, M. H. (2022). Value-driven IT project portfolio management: Process model, evaluation framework, and decision support. *ICIS 2022 Proceedings*, 9. https://aisel.aisnet.org/icis2022/is_design/is_design/9

## Author

Julien Aldag · M.Sc. student in Industrial Engineering and Management (Wirtschaftsingenieurwesen), Leibniz Universität Hannover · [LinkedIn](https://www.linkedin.com/in/julien-aldag)

## License

- **Code:** MIT, see `LICENSE`
- **Dataset:** CDLA-Permissive-1.0, see `LIZENZ_daten.md`; created by ka66ledata ([Kaggle](https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw/data))
