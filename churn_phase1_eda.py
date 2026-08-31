"""
Phase 1 - Data Loading & Understanding
Customer Churn Prediction (Telco Customer Churn dataset)
"""

import pandas as pd

# 1. Load the CSV
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Update the path below to wherever you saved the file.
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

# 2. Shape, columns, dtypes, first 5 rows
print("=" * 60)
print("SHAPE (rows, columns):", df.shape)

print("=" * 60)
print("COLUMN NAMES:")
print(df.columns.tolist())

print("=" * 60)
print("DATA TYPES:")
print(df.dtypes)

print("=" * 60)
print("FIRST 5 ROWS:")
print(df.head())

# 3. Summary statistics for numerical columns
print("=" * 60)
print("SUMMARY STATISTICS (numerical columns):")
print(df.describe())

# 4. Missing values per column
print("=" * 60)
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# 5. Class balance of target column "Churn"
print("=" * 60)
print("CLASS BALANCE OF 'Churn' (counts):")
print(df["Churn"].value_counts())
print("\nCLASS BALANCE OF 'Churn' (percentages):")
print(df["Churn"].value_counts(normalize=True) * 100)
