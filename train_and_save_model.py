"""
Trains the final XGBoost churn pipeline and saves it to disk so the
Streamlit app doesn't need to retrain on every run.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

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

# Save the fitted pipeline (preprocessing + model together) and the
# raw column lists the app's form needs to know about.
joblib.dump(model, "churn_model.joblib")
joblib.dump({"categorical_cols": categorical_cols, "numeric_cols": numeric_cols,
             "X_columns": X.columns.tolist()}, "churn_model_meta.joblib")

print("Saved churn_model.joblib and churn_model_meta.joblib")
