"""
generate_synthetic_data.py

NOTE: This script generates SYNTHETIC data for demonstration purposes.
The real Telco Customer Churn dataset only contains account-level
snapshot data (contract, charges, services) - it does NOT include any
month-by-month usage history or customer support ticket text. To build
and demo a usage-trend (deep learning) layer and an NLP sentiment layer
on top of the existing churn model, this script fabricates both,
correlated with each customer's REAL churn status and MonthlyCharges so
the synthetic signals behave the way real ones plausibly would (not
random noise) - but it is not real usage or support data.

Generates:
- usage_history.csv   : customerID + 5 monthly usage columns
- support_tickets.csv : customerID + ticket_number + ticket_text
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

CLEANED_CSV = "telco_churn_cleaned.csv"
df = pd.read_csv(CLEANED_CSV)

# ---------------------------------------------------------------
# 1. Synthetic 5-month usage history
# ---------------------------------------------------------------
# Baseline usage scales with MonthlyCharges (heavier plans -> more
# usage). Churned customers get a declining trend over the 5 months;
# retained customers get a flat-to-growing trend. Noise is added so
# not every sequence is a perfectly clean line.
n = len(df)
baseline = 200 + df["MonthlyCharges"].to_numpy() * 4  # e.g. $70/mo -> ~480 baseline minutes

usage_cols = [f"month_{i}_usage" for i in range(1, 6)]
usage = np.zeros((n, 5))

for i in range(n):
    is_churn = df["Churn"].iloc[i] == "Yes"
    b = baseline[i]
    if is_churn:
        # Declining trend: total drop of 25-55% across 5 months
        total_drop_frac = rng.uniform(0.25, 0.55)
        trend = np.linspace(1.0, 1.0 - total_drop_frac, 5)
    else:
        # Stable-to-growing: flat, or up to +20% growth
        total_growth_frac = rng.uniform(-0.05, 0.20)
        trend = np.linspace(1.0, 1.0 + total_growth_frac, 5)
    noise = rng.normal(0, 0.06, 5)  # +/-6% month-to-month noise
    usage[i] = np.clip(b * trend * (1 + noise), 10, None)

usage_df = pd.DataFrame(np.round(usage, 1), columns=usage_cols)
usage_df.insert(0, "customerID", df["customerID"].values)
usage_df.to_csv("usage_history.csv", index=False)

# ---------------------------------------------------------------
# 2. Synthetic support ticket text (1-3 tickets per customer)
# ---------------------------------------------------------------
NEGATIVE_TEMPLATES = [
    "My bill went up again for no reason, this is the third time this has happened.",
    "The internet keeps dropping every night, I've called twice and nothing has been fixed.",
    "Nobody responded to my last complaint, I am extremely frustrated with this service.",
    "Overcharged on my last invoice and support was completely unhelpful.",
    "The technician never showed up for the scheduled appointment, very disappointing.",
    "This is way too expensive for the constant connection issues I keep having.",
    "I've been on hold for over an hour and still no resolution to my problem.",
    "Cancelling soon if this billing error isn't fixed immediately.",
]
NEUTRAL_TEMPLATES = [
    "Just calling to update my mailing address on file.",
    "Wanted to confirm my current plan details and billing date.",
    "Asking about upgrading my internet speed for the next cycle.",
    "Quick question about how to set up paperless billing.",
    "Checking if a new streaming add-on is available in my area.",
]
POSITIVE_TEMPLATES = [
    "Thanks for the quick fix last week, the connection has been great since.",
    "Really happy with the service, the new plan works perfectly for us.",
    "The support agent was super helpful and resolved my issue in minutes.",
    "Great experience setting up the new device, very smooth process.",
    "Appreciate the fast response time, everything is working well now.",
]

tickets = []
for _, row in df.iterrows():
    is_churn = row["Churn"] == "Yes"
    n_tickets = rng.integers(1, 4)  # 1-3 tickets
    for t in range(1, n_tickets + 1):
        if is_churn:
            # Mostly negative, occasionally neutral
            text = rng.choice(NEGATIVE_TEMPLATES) if rng.random() < 0.8 else rng.choice(NEUTRAL_TEMPLATES)
        else:
            # Mostly neutral/positive, rarely negative
            roll = rng.random()
            if roll < 0.5:
                text = rng.choice(NEUTRAL_TEMPLATES)
            elif roll < 0.85:
                text = rng.choice(POSITIVE_TEMPLATES)
            else:
                text = rng.choice(NEGATIVE_TEMPLATES)
        tickets.append({"customerID": row["customerID"], "ticket_number": t, "ticket_text": text})

tickets_df = pd.DataFrame(tickets)
tickets_df.to_csv("support_tickets.csv", index=False)

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print(f"Generated usage_history.csv: {len(usage_df)} customers x 5 months")
print(f"Generated support_tickets.csv: {len(tickets_df)} tickets across {tickets_df['customerID'].nunique()} customers")
print()
print("Sample usage_history.csv rows:")
print(usage_df.head(3).to_string(index=False))
print()
print("Sample support_tickets.csv rows:")
print(tickets_df.head(5).to_string(index=False))
