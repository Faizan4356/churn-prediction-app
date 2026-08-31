"""
Phase 2 - Data Cleaning
Customer Churn Prediction (Telco Customer Churn dataset)

Known issue going in: TotalCharges is stored as text (object dtype) and
contains some blank strings (" ") for customers with tenure == 0
(brand-new customers who haven't been billed yet).
"""

import pandas as pd

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("BEFORE CLEANING - info():")
df.info()

# ---------------------------------------------------------------
# 1. Convert TotalCharges to numeric, handling blank/invalid values
# ---------------------------------------------------------------
# errors="coerce" turns blank strings / anything non-numeric into NaN
# so we can see and handle them explicitly instead of them silently
# breaking arithmetic later.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("=" * 60)
print("Rows where TotalCharges became NaN after coercion:")
print(df[df["TotalCharges"].isnull()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

print("=" * 60)
print("AFTER TotalCharges CONVERSION - info():")
df.info()

# ---------------------------------------------------------------
# 2. Handle missing values (reasoning per column)
# ---------------------------------------------------------------
# TotalCharges: every NaN here corresponds to tenure == 0, i.e. a
# customer who just signed up and hasn't been charged yet. The
# correct value isn't "unknown" - it's genuinely 0, since they have
# no billing history. Imputing with mean/median would fabricate a
# billing history that doesn't exist, so we fill with 0.
zero_tenure_mask = df["tenure"] == 0
assert df.loc[df["TotalCharges"].isnull()].index.equals(
    df.loc[zero_tenure_mask & df["TotalCharges"].isnull()].index
), "Found NaN TotalCharges rows that do NOT have tenure == 0 - investigate before filling with 0."

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# All other columns in this dataset are typically complete (no NaNs).
# If your copy has missing values elsewhere, the reasoning is usually:
#   - Numeric (tenure, MonthlyCharges): impute with median (robust to
#     outliers), not mean.
#   - Categorical (Contract, PaymentMethod, etc.): impute with the mode,
#     or add an explicit "Unknown" category if missingness itself might
#     be predictive (e.g. a customer who skipped a survey field).
#   - Never impute the target column ("Churn") - drop those rows instead,
#     since a guessed label would corrupt training.
remaining_na = df.isnull().sum()
remaining_na = remaining_na[remaining_na > 0]
print("=" * 60)
print("Remaining missing values after TotalCharges fix:")
print(remaining_na if not remaining_na.empty else "None")

print("=" * 60)
print("AFTER MISSING VALUE HANDLING - info():")
df.info()

# ---------------------------------------------------------------
# 3. Remove or flag duplicate customer IDs
# ---------------------------------------------------------------
dup_mask = df.duplicated(subset="customerID", keep=False)
print("=" * 60)
print(f"Duplicate customerID rows found: {dup_mask.sum()}")
if dup_mask.any():
    print(df[dup_mask].sort_values("customerID"))

# customerID should be a unique key - keep the first occurrence and
# drop the rest, since duplicate IDs almost always mean a duplicated
# record (e.g. re-exported/merged data) rather than two real customers.
before_rows = len(df)
df = df.drop_duplicates(subset="customerID", keep="first")
print(f"Dropped {before_rows - len(df)} duplicate row(s) based on customerID.")

print("=" * 60)
print("AFTER DUPLICATE REMOVAL - info():")
df.info()

# ---------------------------------------------------------------
# 4. Standardize categorical text values
# ---------------------------------------------------------------
# Trim stray whitespace and unify casing so "Yes " and "yes" aren't
# treated as different categories downstream.
categorical_cols = df.select_dtypes(include="object").columns.drop("customerID")
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

# Several columns (e.g. OnlineSecurity, OnlineBackup, DeviceProtection,
# TechSupport, StreamingTV, StreamingMovies) use "No internet service" or
# "No phone service" as a stand-in for "No" plus an implicit reason.
# For modeling purposes, that reason is fully captured by the
# InternetService/PhoneService columns, so collapsing these to a plain
# "No" removes redundant categories without losing information, and
# keeps the feature strictly binary (Yes/No) which is easier to encode.
collapse_map = {"No internet service": "No", "No phone service": "No"}
for col in categorical_cols:
    df[col] = df[col].replace(collapse_map)

print("=" * 60)
print("Unique values per categorical column after standardization:")
for col in categorical_cols:
    print(f"  {col}: {sorted(df[col].unique())}")

print("=" * 60)
print("FINAL CLEANED DATAFRAME - info():")
df.info()

# Save cleaned data for the next phase.
df.to_csv("telco_churn_cleaned.csv", index=False)
print("=" * 60)
print("Saved cleaned data to telco_churn_cleaned.csv")
