# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers
are likely to cancel their service, explains *why* using SHAP, and serves
live predictions through an interactive Streamlit web app.

**Live demo:** [churn-prediction-app-8ioaenzecxexkcskdmzvxc.streamlit.app](https://churn-prediction-app-8ioaenzecxexkcskdmzvxc.streamlit.app/)

## Overview

This project walks the full lifecycle of a real classification problem:
raw data → cleaning → exploratory analysis → feature engineering → model
comparison → interpretation → a deployed, user-facing app. It's built as a
portfolio piece to demonstrate practical, business-aware machine learning —
not just model accuracy, but *why* the model makes the calls it does and
what a business should do about it.

## Business Problem

Acquiring a new telecom customer costs far more than retaining an existing
one, but a retention team can't call every customer. This project scores
each customer's churn risk and explains the top contributing factors, so
limited retention budget (discounts, contract offers, proactive outreach)
can be targeted at the customers who are both **likely to leave** and
**worth trying to save**.

Because a missed churner (lost recurring revenue) is costlier than a false
alarm (an unnecessary retention offer), the project explicitly optimizes for
**recall** on the churn class rather than raw accuracy — see
[Model Performance](#model-performance) below.

## Dataset

[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(via Kaggle / IBM sample data) — **7,043 customers, 21 features** covering
demographics, account details (contract, tenure, billing), and subscribed
services (phone, internet, streaming, security add-ons).

## Tools & Libraries

| Category | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Visualization | matplotlib, seaborn, Plotly (interactive charts in-app) |
| Model interpretation | SHAP |
| App / deployment | Streamlit, Streamlit Community Cloud |
| Persistence | joblib |

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | `churn_phase1_eda.py` | Initial data loading, shape, dtypes, missing values, class balance |
| 2 | `churn_phase2_cleaning.py` | Fixes `TotalCharges` dtype, handles missing values, dedupes, standardizes categories |
| 3 | `churn_phase3_eda.py` | Visual EDA — churn by contract, tenure, correlation heatmap, charges distribution |
| 4 | `churn_phase4_features.py` | Feature engineering (tenure bins, service count, spend ratio, contract-term flag) |
| 5 | `churn_phase5_models.py` | Trains & compares Logistic Regression, Random Forest, XGBoost |
| 6 | `churn_phase6_evaluation.py` | ROC/AUC, feature importance, SHAP explanations |
| 7 | `app.py` | Streamlit app serving live predictions with charts |

## Key Findings

- **Contract type is the strongest churn driver.** Month-to-month customers
  churn at **42.7%**, versus **11.3%** for one-year contracts and just
  **2.8%** for two-year contracts — an ~15x gap between the least and most
  committed customers.
- **Churn is front-loaded.** Customers with 0–12 months of tenure churn at
  **47.4%**, dropping to **9.5%** for customers past 49 months. Over **55%**
  of all churn happens within a customer's first year.
- **Price sensitivity is real.** Churned customers have a higher median
  monthly bill (**$79.65**) than retained customers (**$64.43**), pointing
  to a value gap on expensive plans.

*(See `churn_phase3_eda.py` for the full charts and per-chart interpretation.)*

## Model Performance

Evaluated on a stratified 80/20 train/test split, with class weighting to
correct for the dataset's ~73/27 class imbalance:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 74.2% | 50.9% | **79.1%** | 0.620 |
| Random Forest | 76.3% | 54.6% | 63.6% | 0.588 |
| **XGBoost (deployed)** | 75.1% | 52.1% | 77.3% | **0.622** |

**XGBoost AUC: 0.839** — meaning the model reliably ranks likely churners
above likely stayers across virtually any decision threshold.

XGBoost was chosen for deployment over Random Forest despite similar
accuracy because it recalls **77.3%** of actual churners versus Random
Forest's 63.6% — given that missing a churner is the costlier mistake for
this business problem, the model that catches more of them wins even at a
similar F1.

**Top predictors** (via feature importance + SHAP): **Contract type**
(by far the largest driver), **Internet service type** (Fiber optic
customers churn more), and **tenure** — confirming the EDA findings above.
Customers who are new, on month-to-month contracts, and on fiber internet
are the highest-risk segment.

## The App

The Streamlit app (`app.py`) lets you enter a customer's details and get:
- A **gauge chart** showing churn probability, color-coded green/amber/red
  by risk level.
- A **SHAP bar chart** of the top 5 factors pushing that specific
  prediction up (red) or down (green).
- A **comparison chart** of the customer's tenure and monthly charges
  against the dataset average.

## Running Locally

**Requirements:** Python 3.10+ and the packages in `requirements.txt`
(Streamlit, pandas, numpy, scikit-learn, XGBoost, SHAP, Plotly, joblib).

```bash
git clone https://github.com/Faizan4356/churn-prediction-app.git
cd churn-prediction-app
pip install -r requirements.txt
```

Download the [Telco Customer Churn CSV](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it in this folder as `WA_Fn-UseC_-Telco-Customer-Churn.csv`, then:

```bash
# Run the pipeline, in order
python churn_phase1_eda.py
python churn_phase2_cleaning.py
python churn_phase3_eda.py
python churn_phase4_features.py
python churn_phase5_models.py
python churn_phase6_evaluation.py

# Train and save the model the app uses
python train_and_save_model.py

# Launch the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

> The repo already ships with a trained `churn_model.joblib` /
> `churn_model_meta.joblib`, so you can skip straight to `streamlit run app.py`
> if you just want to try the app without re-running the pipeline.

## Project Structure

```
├── churn_phase1_eda.py          # Data loading & understanding
├── churn_phase2_cleaning.py     # Data cleaning
├── churn_phase3_eda.py          # Exploratory data analysis
├── churn_phase4_features.py     # Feature engineering
├── churn_phase5_models.py       # Model training & comparison
├── churn_phase6_evaluation.py   # Evaluation & SHAP interpretation
├── train_and_save_model.py      # Trains & serializes the final model
├── app.py                       # Streamlit app
├── churn_model.joblib           # Serialized trained pipeline
├── churn_model_meta.joblib      # Column metadata + averages for the app
└── requirements.txt
```
