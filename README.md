# Customer Churn Prediction

A machine learning pipeline that predicts which telecom customers are likely to
churn, with an interactive Streamlit app for real-time predictions and
explanations.

**Live demo:** [add your Streamlit Community Cloud URL here]

## Business Problem

Acquiring a new customer costs significantly more than retaining an existing
one, but retention teams can't proactively reach out to every customer. This
project builds a model that scores each customer's churn risk and explains
*why*, so retention efforts (discounts, contract offers, outreach) can be
targeted at the customers most likely to leave — and most likely to be saved.

Given that a missed churner (lost revenue) is costlier than a false alarm
(an unnecessary retention offer), the project explicitly optimizes for
**recall** on the churn class rather than raw accuracy.

## Dataset

[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(via Kaggle) — 7,043 customers, 21 features covering demographics, account
details (contract, tenure, billing), and subscribed services.

## Tools & Libraries

- **Data & modeling:** pandas, numpy, scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Model interpretation:** SHAP
- **Deployment:** Streamlit, Streamlit Community Cloud

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | `churn_phase1_eda.py` | Initial data loading, shape, dtypes, missing values, class balance |
| 2 | `churn_phase2_cleaning.py` | Fixes `TotalCharges` dtype, handles missing values, dedupes, standardizes categories |
| 3 | `churn_phase3_eda.py` | Visual EDA — churn by contract, tenure, correlation heatmap, charges distribution |
| 4 | `churn_phase4_features.py` | Feature engineering (tenure bins, service count, spend ratio, contract-term flag) |
| 5 | `churn_phase5_models.py` | Trains & compares Logistic Regression, Random Forest, XGBoost |
| 6 | `churn_phase6_evaluation.py` | ROC/AUC, feature importance, SHAP explanations |
| 7 | `app.py` | Streamlit app serving live predictions |

## Key Findings

- Month-to-month customers churn at **[XX]%** vs. **[XX]%** for two-year
  contracts — the single strongest churn driver in the dataset.
- Churn is front-loaded: **[XX]%** of churn happens within the first 12
  months of tenure, pointing to an onboarding/early-experience problem.
- Churned customers pay a higher median monthly bill (**$[XX]** vs.
  **$[XX]**), suggesting price sensitivity on premium plans.

*(See `churn_phase3_eda.py` output and `plot1`–`plot4` for the full charts
and per-chart interpretation.)*

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | [XX]% | [XX]% | [XX]% | [XX] | [XX] |
| Random Forest | [XX]% | [XX]% | [XX]% | [XX] | [XX] |
| **XGBoost (deployed)** | **[XX]%** | **[XX]%** | **[XX]%** | **[XX]** | **[XX]** |

**Top 3 predictors** (via SHAP): contract type, tenure, and monthly charges —
customers who are new, on month-to-month plans, and paying above-average
bills are the highest-risk segment.

*(Run `churn_phase5_models.py` and `churn_phase6_evaluation.py` on the real
dataset to populate these numbers.)*

## Running Locally

```bash
git clone https://github.com/Faizan4356/churn-prediction-app.git
cd churn-prediction-app
pip install -r requirements.txt

# Run the pipeline (in order) on your copy of the Telco CSV
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
├── churn_model_meta.joblib      # Column metadata for the app
└── requirements.txt
```
