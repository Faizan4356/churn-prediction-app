"""
test_customer_intelligence_platform.py

Run this AFTER completing all 8 build steps to verify everything works
correctly end-to-end before you deploy or show this to anyone.

Usage:
    python test_customer_intelligence_platform.py

This checks each phase's output file exists, has the right shape/columns,
and that the combined pipeline produces sensible results - not just that
the code runs without crashing, but that the outputs make logical sense.

NOTE ON STEP 2: the original version of this test checked for a Keras
usage_trend_model.h5 file. This project's usage-trend model was built in
PyTorch instead of TensorFlow/Keras, because TensorFlow has no published
build for Python 3.14 (this project's interpreter) - see the README and
train_usage_trend_model.py for details. The Step 2 section below has been
adjusted to load usage_trend_model.pt via PyTorch instead; every other
check is unchanged from the original spec.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# ANSI colors for readable terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

results = {"passed": 0, "failed": 0, "warnings": 0}


def check(condition, pass_msg, fail_msg, warning=False):
    """Print a pass/fail line and track results."""
    if condition:
        print(f"{GREEN}[PASS]{RESET} {pass_msg}")
        results["passed"] += 1
    else:
        if warning:
            print(f"{YELLOW}[WARN]{RESET} {fail_msg}")
            results["warnings"] += 1
        else:
            print(f"{RED}[FAIL]{RESET} {fail_msg}")
            results["failed"] += 1


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ----------------------------------------------------------------------
# STEP 1 CHECK: Synthetic data generation
# ----------------------------------------------------------------------
section("STEP 1: Synthetic Data Generation")

check(
    os.path.exists("usage_history.csv"),
    "usage_history.csv exists",
    "usage_history.csv NOT FOUND - run generate_synthetic_data.py first",
)
check(
    os.path.exists("support_tickets.csv"),
    "support_tickets.csv exists",
    "support_tickets.csv NOT FOUND - run generate_synthetic_data.py first",
)

if os.path.exists("usage_history.csv"):
    usage_df = pd.read_csv("usage_history.csv")
    check(
        "customerID" in usage_df.columns,
        "usage_history.csv has customerID column",
        "usage_history.csv is MISSING customerID column",
    )
    month_cols = [c for c in usage_df.columns if c != "customerID"]
    check(
        len(month_cols) == 5,
        f"usage_history.csv has 5 monthly usage columns (found {len(month_cols)})",
        f"Expected 5 monthly usage columns, found {len(month_cols)}",
    )
    check(
        usage_df[month_cols].isnull().sum().sum() == 0,
        "No missing values in usage history",
        "Usage history contains missing values - check generation logic",
        warning=True,
    )
    check(
        (usage_df[month_cols] >= 0).all().all(),
        "All usage values are non-negative (realistic)",
        "Found negative usage values - this is not physically realistic",
    )

if os.path.exists("support_tickets.csv"):
    tickets_df = pd.read_csv("support_tickets.csv")
    check(
        "customerID" in tickets_df.columns and "ticket_text" in tickets_df.columns,
        "support_tickets.csv has required columns",
        "support_tickets.csv is missing customerID or ticket_text column",
    )
    check(
        tickets_df["ticket_text"].str.len().min() > 5,
        "Ticket texts are non-trivial length",
        "Some ticket texts are suspiciously short/empty",
        warning=True,
    )


# ----------------------------------------------------------------------
# STEP 2 CHECK: Usage trend model (PyTorch LSTM - see NOTE at top of file)
# ----------------------------------------------------------------------
section("STEP 2: Usage Trend Deep Learning Model")

check(
    os.path.exists("usage_trend_model.pt"),
    "usage_trend_model.pt exists",
    "usage_trend_model.pt NOT FOUND - run train_usage_trend_model.py first",
)
check(
    os.path.exists("usage_trend_labels.joblib"),
    "usage_trend_labels.joblib exists",
    "usage_trend_labels.joblib NOT FOUND",
)

if os.path.exists("usage_trend_model.pt") and os.path.exists("usage_trend_labels.joblib"):
    try:
        import torch
        from usage_trend_model_def import UsageTrendLSTM

        label_encoder = joblib.load("usage_trend_labels.joblib")
        model = UsageTrendLSTM(input_size=1, hidden_size=16, num_classes=len(label_encoder.classes_))
        model.load_state_dict(torch.load("usage_trend_model.pt"))
        model.eval()
        check(True, "usage_trend_model.pt loads without error", "")

        # Run a sanity prediction on a clearly declining sequence,
        # normalized the same way as during training (per-sequence
        # mean/std) before reshaping to (batch, seq_len, features).
        declining_seq = np.array([100, 80, 60, 40, 20], dtype=np.float32)
        seq_norm = (declining_seq - declining_seq.mean()) / (declining_seq.std() + 1e-6)
        x = torch.tensor(seq_norm.reshape(1, 5, 1), dtype=torch.float32)
        try:
            with torch.no_grad():
                pred_logits = model(x)
            pred_label = label_encoder.inverse_transform([pred_logits.argmax(dim=1).item()])[0]
            check(
                pred_logits is not None,
                "Model produces a prediction on a test declining sequence",
                "",
            )
            check(
                pred_label == "Declining",
                f"Model correctly classifies an obviously declining sequence as 'Declining' (got '{pred_label}')",
                f"Model classified an obviously declining sequence as '{pred_label}' instead of 'Declining' - worth a look",
                warning=True,
            )
        except Exception as e:
            check(
                False,
                "",
                f"Model loaded but failed to predict - check input shape. Error: {e}",
            )
    except Exception as e:
        check(False, "", f"Failed to load usage_trend_model.pt - Error: {e}")


# ----------------------------------------------------------------------
# STEP 3 CHECK: Sentiment/NLP model output
# ----------------------------------------------------------------------
section("STEP 3: NLP Sentiment Analysis")

check(
    os.path.exists("customer_sentiment_scores.csv"),
    "customer_sentiment_scores.csv exists",
    "customer_sentiment_scores.csv NOT FOUND - run train_sentiment_model.py first",
)

if os.path.exists("customer_sentiment_scores.csv"):
    sentiment_df = pd.read_csv("customer_sentiment_scores.csv")
    required_cols = {"customerID", "sentiment_score", "complaint_category"}
    check(
        required_cols.issubset(sentiment_df.columns),
        "customer_sentiment_scores.csv has all required columns",
        f"Missing columns. Expected {required_cols}, found {set(sentiment_df.columns)}",
    )
    check(
        sentiment_df["sentiment_score"].between(-1, 1).all(),
        "All sentiment scores are within expected -1 to 1 range",
        "Some sentiment scores are outside the expected -1 to 1 range - check scoring logic",
        warning=True,
    )
    valid_categories = {"Billing", "Technical", "Service Quality", "Other"}
    unexpected = set(sentiment_df["complaint_category"].unique()) - valid_categories
    check(
        len(unexpected) == 0,
        "All complaint categories match expected values",
        f"Found unexpected complaint categories: {unexpected}",
        warning=True,
    )


# ----------------------------------------------------------------------
# STEP 4 CHECK: Combined health score logic
# ----------------------------------------------------------------------
section("STEP 4: Combined Health Score")

check(
    os.path.exists("customer_health_scores.csv"),
    "customer_health_scores.csv exists",
    "customer_health_scores.csv NOT FOUND - run combine_health_score.py first",
)

if os.path.exists("customer_health_scores.csv"):
    health_df = pd.read_csv("customer_health_scores.csv")
    required_cols = {
        "customerID",
        "churn_risk_score",
        "usage_trend",
        "sentiment_score",
        "health_score",
        "risk_tier",
    }
    check(
        required_cols.issubset(health_df.columns),
        "customer_health_scores.csv has all required columns",
        f"Missing columns. Expected {required_cols}, found {set(health_df.columns)}",
    )
    check(
        health_df["health_score"].between(0, 100).all(),
        "All health scores are within 0-100 range",
        "Some health scores fall outside the 0-100 range - check combine_scores() logic",
    )
    valid_tiers = {"Critical", "At-Risk", "Healthy"}
    check(
        set(health_df["risk_tier"].unique()).issubset(valid_tiers),
        "All risk tiers match expected labels",
        f"Found unexpected risk tier values: {set(health_df['risk_tier'].unique()) - valid_tiers}",
    )

    # Logical sanity check: high churn risk + declining usage + negative
    # sentiment should generally correlate with lower health scores
    if "churn_risk_score" in health_df.columns:
        high_risk_mask = (
            (health_df["churn_risk_score"] > 0.7)
            & (health_df["usage_trend"] == "Declining")
            & (health_df["sentiment_score"] < -0.3)
        )
        if high_risk_mask.sum() > 0:
            avg_score_high_risk = health_df.loc[high_risk_mask, "health_score"].mean()
            avg_score_overall = health_df["health_score"].mean()
            check(
                avg_score_high_risk < avg_score_overall,
                f"High-risk customers score lower on average "
                f"({avg_score_high_risk:.1f} vs overall {avg_score_overall:.1f}) - logic checks out",
                f"High-risk customers do NOT score lower on average "
                f"({avg_score_high_risk:.1f} vs overall {avg_score_overall:.1f}) - "
                f"review your weighting formula in combine_scores()",
            )
        else:
            print(
                f"{YELLOW}[SKIP]{RESET} No customers matched the 'high risk on all "
                f"3 signals' test case - can't verify weighting logic this way"
            )


# ----------------------------------------------------------------------
# STEP 5-7 CHECK: Frontend files exist and reference required functions
# ----------------------------------------------------------------------
section("STEP 5-7: Frontend (app.py) Structure")

check(
    os.path.exists("app.py"),
    "app.py exists",
    "app.py NOT FOUND",
)

if os.path.exists("app.py"):
    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()

    check(
        "st.tabs" in app_code,
        "app.py uses st.tabs() for multi-section layout",
        "app.py does not appear to use st.tabs() - the 3 frontend modes may not be organized correctly",
        warning=True,
    )
    check(
        "Search" in app_code,
        "app.py contains a Search section",
        "Could not find 'Search' reference in app.py - Step 5 may be incomplete",
    )
    check(
        "What-If" in app_code or "What If" in app_code or "Simulator" in app_code,
        "app.py contains a What-If Simulator section",
        "Could not find What-If Simulator reference in app.py - Step 6 may be incomplete",
    )
    check(
        "Batch" in app_code or "upload" in app_code.lower(),
        "app.py contains a Batch Upload section",
        "Could not find Batch Upload reference in app.py - Step 7 may be incomplete",
    )
    check(
        "download" in app_code.lower(),
        "app.py includes a download/export feature for batch results",
        "No download feature found for batch upload results",
        warning=True,
    )


# ----------------------------------------------------------------------
# STEP 8 CHECK: Documentation
# ----------------------------------------------------------------------
section("STEP 8: Documentation")

check(
    os.path.exists("README.md"),
    "README.md exists",
    "README.md NOT FOUND",
)

if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read().lower()
    check(
        "synthetic" in readme,
        "README honestly discloses synthetic data usage",
        "README does NOT mention synthetic data - add this disclosure for honesty",
        warning=True,
    )
    check(
        "lstm" in readme or "deep learning" in readme,
        "README mentions the deep learning layer",
        "README does not mention the deep learning/LSTM component",
        warning=True,
    )
    check(
        "nlp" in readme or "sentiment" in readme,
        "README mentions the NLP layer",
        "README does not mention the NLP/sentiment component",
        warning=True,
    )


# ----------------------------------------------------------------------
# END-TO-END SMOKE TEST: Simulate what the What-If Simulator does
# ----------------------------------------------------------------------
section("END-TO-END: Simulated What-If Prediction")

try:
    if os.path.exists("churn_model.joblib"):
        churn_model = joblib.load("churn_model.joblib")
        check(True, "Existing churn_model.joblib loads correctly", "")
    else:
        check(False, "", "churn_model.joblib not found - original model missing!")
except Exception as e:
    check(False, "", f"Failed to load churn_model.joblib - Error: {e}")

print(
    f"\n{YELLOW}Note:{RESET} This smoke test only checks that the original churn "
    f"model still loads. To fully test a live What-If prediction, manually run "
    f"the Streamlit app and try the simulator with a clearly bad-looking customer "
    f"(month-to-month, high charges, declining usage, angry message) and confirm "
    f"it returns a 'Critical' tier, then try a clearly good customer and confirm "
    f"it returns 'Healthy'."
)


# ----------------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------------
section("TEST SUMMARY")
total = results["passed"] + results["failed"]
print(f"Passed:   {GREEN}{results['passed']}{RESET}")
print(f"Failed:   {RED}{results['failed']}{RESET}")
print(f"Warnings: {YELLOW}{results['warnings']}{RESET}")

if results["failed"] == 0:
    print(
        f"\n{GREEN}All critical checks passed. Review any warnings above, "
        f"then manually test the live app before considering this done.{RESET}"
    )
    sys.exit(0)
else:
    print(
        f"\n{RED}{results['failed']} check(s) failed. Fix these before "
        f"deploying or showing this project to anyone.{RESET}"
    )
    sys.exit(1)
