"""
Phase 6 - Model Evaluation & Interpretation
Customer Churn Prediction (Telco Customer Churn dataset)
Best model: XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
import shap

# Palette (consistent with earlier phases): blue = primary series,
# red = churn/critical status, muted grays for chrome.
COLOR_MODEL = "#2a78d6"
COLOR_BASELINE = "#898781"
COLOR_YES = "#d03b3b"
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "font.family": "sans-serif",
})

CSV_PATH = "telco_churn_features.csv"
df = pd.read_csv(CSV_PATH)

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
    ]
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )),
])
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 1. ROC curve + AUC
# ---------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color=COLOR_MODEL, linewidth=2, label=f"XGBoost (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=COLOR_BASELINE, linewidth=1.5, linestyle="--", label="Random guess (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - XGBoost Churn Model")
ax.grid(color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("plot5_roc_curve.png", dpi=150)
plt.show()
plt.close(fig)

print(f"AUC score: {auc:.3f}")
# An AUC well above 0.5 (typically ~0.82-0.85 on this dataset) means the
# model reliably ranks churners as higher-risk than non-churners across
# ALL possible thresholds - useful because it tells you the model has
# real signal independent of whatever cutoff you eventually pick for
# the recall-vs-precision tradeoff discussed in Phase 5.

# ---------------------------------------------------------------
# 2. Feature importance (top 10)
# ---------------------------------------------------------------
feature_names = model.named_steps["prep"].get_feature_names_out()
importances = model.named_steps["clf"].feature_importances_

importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(10)
    .iloc[::-1]  # reverse so the barh plot reads top-to-bottom by rank
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(importance_df["feature"], importance_df["importance"], color=COLOR_MODEL)
ax.set_xlabel("Importance (gain-based)")
ax.set_title("Top 10 Feature Importances - XGBoost")
ax.grid(axis="x", color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("plot6_feature_importance.png", dpi=150)
plt.show()
plt.close(fig)

print("Top 10 features:")
print(importance_df.iloc[::-1].to_string(index=False))

# ---------------------------------------------------------------
# 3. SHAP values for individual predictions
# ---------------------------------------------------------------
X_test_transformed = model.named_steps["prep"].transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

explainer = shap.TreeExplainer(model.named_steps["clf"])
shap_values = explainer(X_test_df)

# Global summary (how each feature pushes predictions up/down across all customers)
shap.summary_plot(shap_values, X_test_df, show=False)
plt.tight_layout()
plt.savefig("plot7_shap_summary.png", dpi=150)
plt.close()

# Pick 3 example customers: one confidently predicted to churn, one
# confidently predicted to stay, and one the model is unsure about -
# these three cases are the most useful to explain to stakeholders.
churn_probs = pd.Series(y_proba, index=X_test.index)
example_ids = [
    churn_probs.idxmax(),                                   # most likely to churn
    churn_probs.idxmin(),                                   # most likely to stay
    (churn_probs - 0.5).abs().idxmin(),                      # most uncertain
]
example_labels = ["Highest churn risk", "Lowest churn risk", "Most uncertain case"]

for label, idx in zip(example_labels, example_ids):
    row_pos = X_test.index.get_loc(idx)
    print("=" * 60)
    print(f"{label} - customerID: {df.loc[idx, 'customerID']}, "
          f"predicted P(churn) = {churn_probs[idx]:.3f}, actual = {y_test[idx]}")
    plt.figure()  # shap.plots.waterfall draws on plt.gcf(), so start from a clean figure
    shap.plots.waterfall(shap_values[row_pos], show=False)
    plt.tight_layout()
    plt.savefig(f"plot8_shap_waterfall_{label.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()
    print(f"Saved waterfall plot for this customer showing which features pushed "
          f"the prediction up (toward churn) vs down (toward staying).")

print("""
WHAT THE TOP 3 FEATURES TELL US (plain language)
=================================================
Exact ranking will vary by run, but on this dataset the top features are
consistently some combination of: Contract type, tenure, and
MonthlyCharges. Here's what each means for the business:

1. Contract type (Month-to-month vs One/Two year)
   Customers with no long-term commitment churn far more, because
   there's no switching cost keeping them - they can leave whenever a
   competitor's offer looks better. This is the most actionable finding:
   converting month-to-month customers to annual contracts (via
   discounts or loyalty perks) directly reduces their ability to churn
   on a whim, not just their desire to.

2. Tenure
   Newer customers churn more. This isn't really about "how long
   they've had the service" - it reflects that the first several months
   are when a customer is still deciding if the service is worth it and
   hasn't yet built habits or switching costs around it. It signals that
   onboarding quality and early customer support have an outsized effect
   on lifetime retention compared to ongoing support later on.

3. MonthlyCharges
   Higher-paying customers churn more, which points to price sensitivity
   or a value gap - customers on expensive plans (often bundled with
   add-ons) may not feel the extra cost is justified. This suggests the
   business should audit whether high-cost customers are actually using
   what they're paying for, and consider proactive plan reviews or
   loyalty pricing before they churn over cost rather than after.

Together, these three features tell a coherent story: churn risk is
highest for customers who are new, uncommitted (month-to-month), and
paying a lot - i.e., customers who haven't yet been "locked in" by
tenure or contract terms, and who feel the price pinch before they've
built loyalty. Retention strategy should target this exact intersection
first.
""")
