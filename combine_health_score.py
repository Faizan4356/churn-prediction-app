"""
combine_health_score.py

Combines the three independent signals - churn risk (ML), usage trend
(DL/LSTM), and sentiment (NLP) - into one 0-100 Health Score per
customer. This is the integration layer: it's what turns three separate
models into one coherent "Customer Intelligence" system.

The scoring functions (combine_scores, assign_tier, generate_explanation)
are also imported directly by app.py's What-If Simulator and Batch
Upload tabs, so this module only runs its data-loading/batch-scoring
logic when executed as a script (`python combine_health_score.py`),
not on import.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# combine_scores() - the weighted formula
# ---------------------------------------------------------------
def combine_scores(churn_risk, usage_trend, sentiment_score):
    """
    Combines churn risk (0-1, higher = worse), usage_trend
    ("Declining"/"Stable"/"Growing"), and sentiment_score (-1 to +1,
    lower = worse) into one 0-100 Health Score, where HIGHER is better
    (matching a "customer health" framing rather than a risk-score
    framing, so 100 = perfectly healthy, 0 = critical).

    Weighting rationale:
    - Churn risk is the base of the score (weighted highest, ~70% of
      the range) because it's a single model trained specifically and
      exclusively to predict this exact outcome (AUC 0.839 on held-out
      data) - it's the most directly predictive signal available.
    - Usage trend and sentiment are treated as MODIFIERS on top of that
      base, each able to shift the score by up to 15 points in either
      direction. They matter, but neither is as strong a churn signal
      on its own as the dedicated churn model, and they can reinforce
      or partially offset the churn score (e.g. a moderate-risk
      customer with a sharply declining usage trend and angry recent
      tickets should end up looking worse than the churn score alone
      suggests; a moderate-risk customer with growing usage and happy
      tickets should look better).
    """
    base = (1 - churn_risk) * 100 * 0.70 + 30  # maps churn_risk [0,1] -> base [30,100], inverted so lower risk = higher base

    usage_modifier = {"Declining": -15, "Stable": 0, "Growing": 15}.get(usage_trend, 0)

    # sentiment_score in [-1, 1] -> modifier in [-15, +15]
    sentiment_modifier = np.clip(sentiment_score, -1, 1) * 15

    score = base + usage_modifier + sentiment_modifier
    return float(np.clip(score, 0, 100))


def assign_tier(health_score):
    if health_score <= 33:
        return "Critical"
    elif health_score <= 66:
        return "At-Risk"
    return "Healthy"


def generate_explanation(row):
    """Plain-language string describing which signals are driving the score."""
    parts = []
    if row["churn_risk_score"] >= 0.66:
        parts.append("high churn risk")
    elif row["churn_risk_score"] >= 0.33:
        parts.append("moderate churn risk")
    else:
        parts.append("low churn risk")

    if row["usage_trend"] == "Declining":
        parts.append("declining usage")
    elif row["usage_trend"] == "Growing":
        parts.append("growing usage")
    else:
        parts.append("stable usage")

    if row["sentiment_score"] < -0.3:
        parts.append("negative recent complaints")
    elif row["sentiment_score"] > 0.3:
        parts.append("positive recent feedback")
    else:
        parts.append("neutral recent contact history")

    return ", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 1 else parts[0]


def main():
    import torch
    import joblib
    from usage_trend_model_def import UsageTrendLSTM

    # ---------------------------------------------------------------
    # 1. Load the three signals
    # ---------------------------------------------------------------
    churn_model = joblib.load("churn_model.joblib")
    churn_meta = joblib.load("churn_model_meta.joblib")

    # telco_churn_features.csv (Phase 4's output) isn't committed to the
    # repo, so rebuild the same engineered columns here from
    # telco_churn_cleaned.csv, keeping this script self-contained.
    cleaned = pd.read_csv("telco_churn_cleaned.csv")

    bins = [-1, 12, 24, 48, cleaned["tenure"].max()]
    labels = ["0-12", "13-24", "25-48", "49+"]
    cleaned["tenure_group"] = pd.cut(cleaned["tenure"], bins=bins, labels=labels)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    cleaned["total_services"] = (cleaned[service_cols] == "Yes").sum(axis=1)
    cleaned["avg_monthly_spend"] = np.where(
        cleaned["tenure"] > 0, cleaned["TotalCharges"] / cleaned["tenure"], cleaned["MonthlyCharges"]
    )
    protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    cleaned["has_protection_addon"] = (cleaned[protection_cols] == "Yes").any(axis=1).astype(int)
    contract_months = cleaned["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24})
    cleaned["within_contract_term"] = (cleaned["tenure"] < contract_months).astype(int)

    X = cleaned[churn_meta["X_columns"]]
    churn_risk_scores = churn_model.predict_proba(X)[:, 1]  # 0-1, higher = more likely to churn

    usage_df = pd.read_csv("usage_history.csv")
    usage_cols = [c for c in usage_df.columns if c.startswith("month_")]
    sentiment_df = pd.read_csv("customer_sentiment_scores.csv")

    # ---------------------------------------------------------------
    # Run the usage-trend LSTM over every customer's sequence
    # ---------------------------------------------------------------
    label_encoder = joblib.load("usage_trend_labels.joblib")
    usage_model = UsageTrendLSTM(input_size=1, hidden_size=16, num_classes=len(label_encoder.classes_))
    usage_model.load_state_dict(torch.load("usage_trend_model.pt"))
    usage_model.eval()

    sequences = usage_df[usage_cols].to_numpy(dtype=np.float32)
    seq_mean = sequences.mean(axis=1, keepdims=True)
    seq_std = sequences.std(axis=1, keepdims=True) + 1e-6
    sequences_norm = ((sequences - seq_mean) / seq_std).reshape(-1, 5, 1)

    with torch.no_grad():
        logits = usage_model(torch.tensor(sequences_norm, dtype=torch.float32))
        trend_preds = logits.argmax(dim=1).numpy()
    usage_df["usage_trend"] = label_encoder.inverse_transform(trend_preds)

    # ---------------------------------------------------------------
    # 2. Apply combine_scores() across all customers
    # ---------------------------------------------------------------
    result = pd.DataFrame({
        "customerID": cleaned["customerID"],
        "churn_risk_score": churn_risk_scores,
    })
    result = result.merge(usage_df[["customerID", "usage_trend"]], on="customerID", how="left")
    result = result.merge(sentiment_df, on="customerID", how="left")

    # Customers with no support tickets (shouldn't happen with this
    # synthetic generator, but guard anyway) get neutral sentiment.
    result["sentiment_score"] = result["sentiment_score"].fillna(0.0)
    result["complaint_category"] = result["complaint_category"].fillna("None")

    result["health_score"] = result.apply(
        lambda row: combine_scores(row["churn_risk_score"], row["usage_trend"], row["sentiment_score"]),
        axis=1,
    )
    result["risk_tier"] = result["health_score"].apply(assign_tier)
    result["explanation"] = result.apply(generate_explanation, axis=1)

    result.to_csv("customer_health_scores.csv", index=False)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("Risk tier distribution:")
    print(result["risk_tier"].value_counts())
    print("\nSample rows:")
    print(result.head(5).to_string(index=False))
    print("\nSaved customer_health_scores.csv")


if __name__ == "__main__":
    main()
