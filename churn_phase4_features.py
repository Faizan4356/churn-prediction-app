"""
Phase 4 - Feature Engineering
Customer Churn Prediction (Telco Customer Churn dataset)

Columns available (post-cleaning):
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn, tenure_group
"""

import pandas as pd
import numpy as np

CSV_PATH = "telco_churn_cleaned.csv"
df = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------
# 1. Tenure grouped into categories
# ---------------------------------------------------------------
# Why: Phase 3's EDA showed churn is heavily front-loaded in the first
# 12 months and then drops off. A tree-based model can find that
# threshold on its own from raw tenure, but linear/logistic models
# can't express a non-linear "new customers are risky" relationship
# from a single continuous number - binning makes that pattern
# explicit and usable by any model type.
bins = [-1, 12, 24, 48, df["tenure"].max()]
labels = ["0-12", "13-24", "25-48", "49+"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

# ---------------------------------------------------------------
# 2. Total services subscribed
# ---------------------------------------------------------------
# Why: Customers who use more of the ecosystem (security, backup,
# streaming, tech support, etc.) are more "locked in" - both because
# switching providers means giving up more, and because heavy usage
# often signals genuine satisfaction/reliance on the service. A single
# count feature captures overall engagement instead of forcing the
# model to learn the same relationship independently across 6+ sparse
# Yes/No columns.
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)

# ---------------------------------------------------------------
# 3. Average monthly spend ratio (TotalCharges / tenure)
# ---------------------------------------------------------------
# Why: TotalCharges and tenure are highly correlated (Phase 3's
# heatmap), so TotalCharges mostly just re-expresses "how long has
# this customer been here." Dividing it by tenure recovers the
# customer's *effective* historical monthly rate, which can catch
# cases where MonthlyCharges (today's rate) diverges from what they've
# actually been paying on average - e.g. after a recent plan change or
# promo expiration. tenure == 0 would cause a division error, so those
# customers (who have no billing history yet) get their current
# MonthlyCharges as the best available estimate.
df["avg_monthly_spend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"],
)

# ---------------------------------------------------------------
# 4. Has multiple/premium add-ons flag
# ---------------------------------------------------------------
# Why: Distinct from raw service count, this flags customers on
# security/protection-type add-ons specifically (as opposed to
# entertainment add-ons like streaming). These tend to be
# reliability/trust-driven purchases rather than lifestyle purchases,
# and may carry different churn behavior - a binary flag lets the
# model treat "protection-oriented" customers as a segment without
# needing to separately weight each individual add-on column.
protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
df["has_protection_addon"] = (df[protection_cols] == "Yes").any(axis=1).astype(int)

# ---------------------------------------------------------------
# 5. Contract-to-tenure mismatch (short tenure on a long contract)
# ---------------------------------------------------------------
# Why: A customer on a One/Two year contract who is still within their
# first term (tenure < contract length) is "locked in" and structurally
# unlikely to churn regardless of satisfaction - their low churn risk
# is partly mechanical, not behavioral. Flagging this separates "safe
# because committed" from "safe because happy," which matters if the
# business wants to identify customers who are actually satisfied
# versus ones who are just contractually stuck (and may churn the
# moment their contract is up).
contract_months = df["Contract"].map({
    "Month-to-month": 1, "One year": 12, "Two year": 24,
})
df["within_contract_term"] = (df["tenure"] < contract_months).astype(int)

# ---------------------------------------------------------------
# Sanity check: preview the new features
# ---------------------------------------------------------------
new_cols = [
    "tenure_group", "total_services", "avg_monthly_spend",
    "has_protection_addon", "within_contract_term",
]
print(df[["customerID", "tenure", "Contract"] + new_cols].head(10))

df.to_csv("telco_churn_features.csv", index=False)
print("\nSaved feature-engineered data to telco_churn_features.csv")
