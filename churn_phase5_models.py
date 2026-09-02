"""
Phase 5 - Model Building & Comparison
Customer Churn Prediction (Telco Customer Churn dataset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

CSV_PATH = "telco_churn_features.csv"
df = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------
# 0. Prep target and drop identifier / leakage-prone columns
# ---------------------------------------------------------------
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])

# tenure_group is a derived duplicate of tenure - keep the engineered
# categorical version and let the model learn the binning itself is
# fine too, but for tree models the raw + binned versions together are
# redundant. We keep both here since tree models handle correlated
# features gracefully, and it costs nothing to let the model choose.

# ---------------------------------------------------------------
# 1. Train/test split (80/20, stratified)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------
# 2 & 3. Encode categoricals + scale numerics via ColumnTransformer
# ---------------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
    ]
)

# ---------------------------------------------------------------
# 4. Train 3 models
# ---------------------------------------------------------------
# class_weight="balanced" / scale_pos_weight compensate for the ~73/27
# class imbalance found in Phase 1, so models aren't biased toward
# blindly predicting "No".
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
    ]),
}

results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

    print("=" * 60)
    print(f"{name}")
    print("=" * 60)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of predicted churners, how many actually churned)")
    print(f"Recall:    {rec:.3f}  (of actual churners, how many we caught)")
    print(f"F1-score:  {f1:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print()
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ---------------------------------------------------------------
# 5. Summary comparison table
# ---------------------------------------------------------------
summary = pd.DataFrame({
    name: {k: v for k, v in res.items() if k != "cm"}
    for name, res in results.items()
}).T
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(summary.round(3))

# ---------------------------------------------------------------
# 5b. Model comparison chart - a grouped bar chart, one color per
# model (identity encoding), grouped by metric so it's easy to see
# which model wins on which axis. Colors are the validated
# categorical palette's first 3 slots (these 3 clear every CVD/
# contrast check together, in this fixed order).
# ---------------------------------------------------------------
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",  # categorical slot 1 (blue)
    "Random Forest": "#eb6834",        # categorical slot 2 (orange)
    "XGBoost": "#1baf7a",              # categorical slot 3 (aqua)
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"

metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
model_names = list(results.keys())

fig, ax = plt.subplots(figsize=(8, 5))
n_models = len(model_names)
bar_width = 0.8 / n_models
x = np.arange(len(metrics))

for i, name in enumerate(model_names):
    values = [results[name][m] for m in metrics]
    offset = (i - (n_models - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values, width=bar_width, label=name, color=MODEL_COLORS[name])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison - Logistic Regression vs Random Forest vs XGBoost")
ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(AXIS_COLOR)
ax.spines["bottom"].set_color(AXIS_COLOR)
ax.tick_params(colors=MUTED_COLOR)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("plot9_model_comparison.png", dpi=150)
plt.show()
print("Saved plot9_model_comparison.png")

# ---------------------------------------------------------------
# 6. Recommendation
# ---------------------------------------------------------------
print("""
RECOMMENDATION
==============
For this business problem, a missed churner (false negative) is far more
costly than a false alarm (false positive): a false alarm just means an
unnecessary retention offer (a small discount or a check-in call), while
a missed churner means losing the customer's full future revenue with no
chance to intervene. That makes RECALL on the "Churn" class the metric to
optimize for, not accuracy.

Given that, the recommended model is whichever of Random Forest / XGBoost
scores highest on recall in the printed results above (run this script to
get numbers on your actual data) - XGBoost is the more likely winner in
practice for tabular churn data of this size, since gradient boosting
typically captures non-linear interactions (e.g. "month-to-month contract
AND high monthly charges AND low tenure") slightly better than a random
forest, and scale_pos_weight lets it directly account for the class
imbalance.

Logistic Regression is still valuable even if it isn't deployed as the
final model: its coefficients are directly interpretable, so it's the
one to hand to the business team to explain WHY a customer is flagged as
at-risk (e.g. "each additional year of contract commitment reduces churn
odds by X%"), which matters for designing retention interventions, not
just predicting they're needed.

In production, don't stop at the default 0.5 probability threshold -
since false negatives are costlier, LOWER the classification threshold
(e.g. flag anyone with predicted churn probability > 0.3) to trade some
precision for higher recall, and validate that tradeoff against the
actual cost of a retention offer vs. the cost of a lost customer.
""")
