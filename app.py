"""
Streamlit app - Customer Intelligence Platform

Run locally with:
    streamlit run app.py

Tab 1 (Single Prediction) requires churn_model.joblib and
churn_model_meta.joblib (produced by train_and_save_model.py).

Tabs 2-4 (Search / What-If / Batch Upload) additionally require:
- usage_trend_model.pt + usage_trend_labels.joblib (train_usage_trend_model.py)
- customer_health_scores.csv (combine_health_score.py) - tab 2 only
Each of those tabs shows a clear message instead of crashing if its
required file is missing, so the original churn predictor keeps working
even if the newer files haven't been generated yet.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import torch
import plotly.graph_objects as go

from usage_trend_model_def import UsageTrendLSTM
from combine_health_score import combine_scores, assign_tier, generate_explanation

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📉", layout="centered")

# ---------------------------------------------------------------
# Color palette - status colors (good/warning/critical) used
# everywhere a risk level is shown, so every chart, badge, and card
# uses the same meaning-to-color mapping instead of picking colors
# ad hoc.
# ---------------------------------------------------------------
COLOR_GOOD = "#0ca30c"       # low risk / retained / healthy
COLOR_WARNING = "#eda100"    # medium risk / at-risk
COLOR_CRITICAL = "#d03b3b"   # high risk / churned / critical
COLOR_BLUE = "#3987e5"       # primary / neutral series (lightened for dark background)
COLOR_MUTED = "#9a9890"      # secondary / "average customer" comparisons
GRID_COLOR = "rgba(255,255,255,0.15)"   # gridlines - a dark-surface hairline, not the light-mode one
TEXT_COLOR = "#e8e8e6"       # chart text/labels/ticks on the dark page background

# ---------------------------------------------------------------
# Plain-language labels used to translate raw column/feature names
# into sentences a non-technical reader can follow.
# ---------------------------------------------------------------
NUMERIC_PHRASES = {
    "tenure": "How long they've been a customer",
    "MonthlyCharges": "How much they pay per month",
    "TotalCharges": "How much they've paid in total",
    "total_services": "How many services they've signed up for",
    "avg_monthly_spend": "Their average monthly spend",
    "has_protection_addon": "Whether they have a security/protection add-on",
    "within_contract_term": "Whether they're still locked into their current contract",
    "SeniorCitizen": "Whether they are a senior citizen",
}
CATEGORY_PHRASES = {
    "Contract": "Contract type",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "gender": "Gender",
    "Partner": "Has a partner",
    "Dependents": "Has dependents",
    "PhoneService": "Has phone service",
    "MultipleLines": "Has multiple phone lines",
    "OnlineSecurity": "Has online security",
    "OnlineBackup": "Has online backup",
    "DeviceProtection": "Has device protection",
    "TechSupport": "Has tech support",
    "StreamingTV": "Streams TV",
    "StreamingMovies": "Streams movies",
    "PaperlessBilling": "Uses paperless billing",
    "tenure_group": "Tenure group",
}


def humanize_feature(token, categorical_cols, numeric_cols):
    """Turn an encoded feature name like 'cat__Contract_Month-to-month'
    into a plain-English label like 'Contract type: Month-to-month'."""
    name = token.split("__", 1)[-1]
    if name in numeric_cols:
        return NUMERIC_PHRASES.get(name, name.replace("_", " "))
    for col in sorted(categorical_cols, key=len, reverse=True):
        prefix = col + "_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            phrase = CATEGORY_PHRASES.get(col, col)
            return f"{phrase}: {value}"
    return name.replace("_", " ")


def engineer_customer_row(
    gender, senior_citizen, partner, dependents, tenure, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges,
    total_charges=None,
):
    """Builds one model-ready input row from raw account fields,
    recreating the same engineered features from Phase 4. Shared by
    every tab that scores an account (Single Prediction, What-If,
    Batch Upload) so the feature logic lives in exactly one place."""
    if total_charges is None:
        total_charges = tenure * monthly_charges

    raw = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    bins = [-1, 12, 24, 48, 200]
    labels = ["0-12", "13-24", "25-48", "49+"]
    raw["tenure_group"] = pd.cut([tenure], bins=bins, labels=labels)[0]

    service_cols_yes = [
        phone_service, multiple_lines, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
    ]
    raw["total_services"] = sum(1 for v in service_cols_yes if v == "Yes")
    raw["avg_monthly_spend"] = total_charges / tenure if tenure > 0 else monthly_charges
    raw["has_protection_addon"] = int(any(
        v == "Yes" for v in [online_security, online_backup, device_protection, tech_support]
    ))
    contract_months_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    raw["within_contract_term"] = int(tenure < contract_months_map[contract])

    return raw, service_cols_yes


def predict_usage_trend(usage_values, usage_model, label_encoder):
    """5 monthly usage numbers -> 'Declining'/'Stable'/'Growing'."""
    seq = np.array(usage_values, dtype=np.float32)
    seq_mean = seq.mean()
    seq_std = seq.std() + 1e-6
    seq_norm = (seq - seq_mean) / seq_std
    x = torch.tensor(seq_norm.reshape(1, 5, 1), dtype=torch.float32)
    with torch.no_grad():
        logits = usage_model(x)
        pred = logits.argmax(dim=1).item()
    return label_encoder.inverse_transform([pred])[0]


def predict_sentiment(texts, sentiment_tools):
    """List of ticket strings -> list of signed sentiment scores in [-1, 1].

    Calls the tokenizer and model directly rather than through
    transformers' high-level pipeline() wrapper - on this environment,
    pipeline()'s internal batching hung indefinitely even on a handful
    of short texts, while direct tokenize -> forward -> softmax calls
    are fast and reliable (see train_sentiment_model.py for the same
    fix and a fuller explanation).

    If the sentiment model failed to load (sentiment_tools is None -
    e.g. this deployment's sandbox blocks the model download and has
    no local cache), returns neutral (0.0) for every text instead of
    crashing, so the rest of the app keeps working.
    """
    if sentiment_tools is None:
        return [0.0] * len(texts)
    tokenizer, sentiment_model = sentiment_tools
    scores = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            logits = sentiment_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            label_id = probs.argmax().item()
            label = sentiment_model.config.id2label[label_id]
            confidence = probs[label_id].item()
            scores.append(confidence if label == "POSITIVE" else -confidence)
    return scores


# ---------------------------------------------------------------
# Ready-made example customers so someone who doesn't know what
# values to enter can just click a button and see how the app works.
# ---------------------------------------------------------------
PRESETS = {
    "loyal": dict(
        gender="Male", senior_citizen="No", partner="Yes", dependents="Yes",
        tenure=60, monthly_charges=45.0, contract="Two year",
        paperless_billing="No", payment_method="Credit card (automatic)",
        phone_service="Yes", multiple_lines="Yes", internet_service="DSL",
        online_security="Yes", online_backup="Yes", device_protection="Yes",
        tech_support="Yes", streaming_tv="No", streaming_movies="No",
    ),
    "average": dict(
        gender="Female", senior_citizen="No", partner="No", dependents="No",
        tenure=24, monthly_charges=65.0, contract="One year",
        paperless_billing="Yes", payment_method="Bank transfer (automatic)",
        phone_service="Yes", multiple_lines="No", internet_service="DSL",
        online_security="No", online_backup="Yes", device_protection="No",
        tech_support="No", streaming_tv="Yes", streaming_movies="No",
    ),
    "at_risk": dict(
        gender="Female", senior_citizen="No", partner="No", dependents="No",
        tenure=2, monthly_charges=95.0, contract="Month-to-month",
        paperless_billing="Yes", payment_method="Electronic check",
        phone_service="Yes", multiple_lines="Yes", internet_service="Fiber optic",
        online_security="No", online_backup="No", device_protection="No",
        tech_support="No", streaming_tv="Yes", streaming_movies="Yes",
    ),
}


def apply_preset(preset_key):
    for field, value in PRESETS[preset_key].items():
        st.session_state[field] = value


# ---------------------------------------------------------------
# Light styling for a more polished look - a colored gradient header
# band and card-style containers. Kept minimal so it still respects
# the viewer's light/dark theme rather than fighting it.
# ---------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .hero-banner {{
        background: linear-gradient(135deg, {COLOR_BLUE} 0%, #4a3aa7 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }}
    .hero-banner h1 {{
        color: white;
        margin: 0 0 0.4rem 0;
        font-size: 1.8rem;
    }}
    .hero-banner p {{
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 0.95rem;
    }}
    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }}
    /* st.metric clips its value with an ellipsis by default when the
       column is narrow (e.g. 3 metrics side by side on a small
       screen) - allow it to wrap onto a second line instead of
       truncating short words like "Neutral" into "Neut...". The
       truncating white-space/overflow rules are actually set on the
       inner <p>, not the stMetricValue div itself, so target both.
       Also shrink the value font so words like "Growing"/"Positive"
       fit on one line in a narrow column instead of splitting
       mid-word ("Growin" / "g") - overflow-wrap is a fallback for the
       rare case a single word still can't fit, not the primary wrap
       mechanism. */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] p {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: break-word;
        line-height: 1.25;
        font-size: 1.4rem !important;
    }}
    /* Black theme - a near-black page with a faint dark-blue/violet
       wash (not flat black) so panels still read as distinct layers,
       and dark semi-transparent "cards" for the form and content
       blocks with a light hairline border for separation. */
    [data-testid="stAppViewContainer"] > .main {{
        background: radial-gradient(circle at 20% 0%, #14182b 0%, #0a0a0d 45%, #050506 100%);
    }}
    div[data-testid="stForm"] {{
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 1.25rem 1.5rem 0.5rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.10);
    }}
    div.block-container div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
    }}
    .health-badge {{
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    </style>
    <div class="hero-banner">
        <h1>📉 Customer Intelligence Platform</h1>
        <p>Predict churn risk, explore usage trends, and read customer
        sentiment - all in one place, explained in plain English.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("❓ New here? Read this first (30 seconds)", expanded=False):
    st.markdown(
        """
        **What this app does:** it combines three separate models into one
        "Customer Health Score":
        1. A **churn model** (XGBoost, trained on 7,043 real customers)
           that predicts how likely someone is to cancel.
        2. A **usage-trend model** (an LSTM neural network) that looks at
           a customer's last 5 months of usage and classifies it as
           Declining, Stable, or Growing.
        3. An **NLP sentiment model** (a pretrained DistilBERT model) that
           reads a customer's recent support messages and scores how
           positive or negative they sound.

        These three signals are combined into one **0-100 Health Score**
        (higher = healthier) with a **Critical / At-Risk / Healthy** tier,
        so a retention team can act on one number instead of three.

        > ⚠️ **Honesty note:** the real Telco dataset only has account-level
        > data (contract, charges, services) - it has no usage history or
        > support tickets. So for tabs 2-4 below, usage history and support
        > messages are **synthetically generated** for demonstration,
        > correlated with each customer's real churn status. The **churn
        > model itself is trained on 100% real data.**

        **The 4 tabs:**
        - **Single Prediction** — the original churn-only predictor.
        - **Search Existing Customer** — look up any of the 7,043 training
          customers by ID and see their full Health Score breakdown.
        - **What-If Simulator** — type in a hypothetical customer's
          details, usage, and a sample support message, and get a live
          Health Score.
        - **Batch Upload** — upload a CSV of many customers and score them
          all at once.
        """
    )

# ---------------------------------------------------------------
# Load model (cached so it's loaded once per session, not per click)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.joblib")
    meta = joblib.load("churn_model_meta.joblib")
    return model, meta


@st.cache_resource
def load_usage_model():
    label_encoder = joblib.load("usage_trend_labels.joblib")
    usage_model = UsageTrendLSTM(input_size=1, hidden_size=16, num_classes=len(label_encoder.classes_))
    usage_model.load_state_dict(torch.load("usage_trend_model.pt"))
    usage_model.eval()
    return usage_model, label_encoder


@st.cache_resource
def load_sentiment_pipeline():
    """Loads the tokenizer + model directly (not transformers'
    pipeline() wrapper - see predict_sentiment's docstring).

    Tries the local cache first (fast, and avoids a slow DNS-retry
    storm some networks hit on the Hub's "is there a newer version"
    check), falls back to a normal online download if nothing is
    cached yet, and returns None if BOTH fail for any reason (blocked
    outbound network, disk/permission limits, out of memory, etc. -
    whatever the specific cause, this deployment's sandbox won't allow
    it) so the caller can degrade gracefully instead of crashing the
    whole app."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception:
            return None
    sentiment_model.eval()
    return tokenizer, sentiment_model


@st.cache_data
def load_health_scores():
    return pd.read_csv("customer_health_scores.csv")


model, meta = load_model()

# ---------------------------------------------------------------
# Shared UI: renders a Health Score breakdown. Used by both the
# Search and What-If tabs so the two look and behave identically.
# ---------------------------------------------------------------
def render_health_breakdown(churn_risk_score, usage_trend, sentiment_score, complaint_category, health_score, risk_tier, explanation):
    tier_color = {"Critical": COLOR_CRITICAL, "At-Risk": COLOR_WARNING, "Healthy": COLOR_GOOD}[risk_tier]

    hcol1, hcol2 = st.columns([1, 2])
    with hcol1:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:3rem; font-weight:700; color:{tier_color}'>{health_score:.0f}</div>"
            f"<div style='color:{COLOR_MUTED}; font-size:0.85rem; margin-bottom:0.5rem'>Health Score (0-100, higher is better)</div>"
            f"<span class='health-badge' style='background:{tier_color}'>{risk_tier}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with hcol2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn risk", f"{churn_risk_score:.0%}")
        m2.metric("Usage trend", usage_trend)
        sentiment_word = "Positive" if sentiment_score > 0.3 else "Negative" if sentiment_score < -0.3 else "Neutral"
        m3.metric("Recent sentiment", sentiment_word)
        st.caption(f"Most common complaint type: **{complaint_category}**")

    st.info(f"**In plain words:** {explanation}.")


# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_predict, tab_search, tab_whatif, tab_batch = st.tabs(
    ["🔮 Single Prediction", "🔍 Search Existing Customer", "🧪 What-If Simulator", "📁 Batch Upload"]
)

# =================================================================
# TAB 1: Single Prediction (the original churn-only predictor)
# =================================================================
with tab_predict:
    st.subheader("📊 What drives churn in general?")
    st.markdown(
        "These two charts summarize patterns found across **all 7,043 "
        "customers** in the training data (not the one you'll enter below). "
        "They explain *why* the model pays so much attention to contract type "
        "and tenure."
    )

    ref_col1, ref_col2 = st.columns(2)

    with ref_col1:
        contract_fig = go.Figure(go.Bar(
            x=["Month-to-\nmonth", "One year", "Two year"],
            y=[42.7, 11.3, 2.8],
            marker_color=[COLOR_CRITICAL, COLOR_WARNING, COLOR_GOOD],
            text=["42.7%", "11.3%", "2.8%"],
            textposition="outside",
        ))
        contract_fig.update_layout(
            title="Churn rate by contract type",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="% who cancelled",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, range=[0, 50]),
        )
        st.plotly_chart(contract_fig, use_container_width=True)
        st.caption(
            "**How to read this:** each bar is a contract type, and its height "
            "is the percentage of customers on that contract who cancelled. "
            "Month-to-month customers cancel **15x more often** than two-year "
            "customers — because they can leave anytime with no penalty, while "
            "longer contracts lock them in. This is the single strongest "
            "pattern in the whole dataset."
        )

    with ref_col2:
        tenure_fig = go.Figure(go.Bar(
            x=["0-12\nmonths", "13-24\nmonths", "25-48\nmonths", "49+\nmonths"],
            y=[47.4, 28.7, 20.4, 9.5],
            marker_color=[COLOR_CRITICAL, COLOR_WARNING, COLOR_WARNING, COLOR_GOOD],
            text=["47.4%", "28.7%", "20.4%", "9.5%"],
            textposition="outside",
        ))
        tenure_fig.update_layout(
            title="Churn rate by how long they've stayed",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="% who cancelled",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, range=[0, 55]),
        )
        st.plotly_chart(tenure_fig, use_container_width=True)
        st.caption(
            "**How to read this:** each bar is a group of customers by how "
            "many months they've stayed. Brand-new customers (under a year) "
            "cancel nearly **5x more often** than long-time customers (past 4 "
            "years). Over half of all cancellations happen in a customer's "
            "first 12 months — so the first year is when a company should "
            "work hardest to keep someone happy."
        )

    st.divider()

    st.subheader("1. Pick an example, or fill in your own")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.button("🟢 Loyal customer example", use_container_width=True,
                  on_click=apply_preset, args=("loyal",), key="preset_loyal_btn")
    with b2:
        st.button("🟡 Average customer example", use_container_width=True,
                  on_click=apply_preset, args=("average",), key="preset_avg_btn")
    with b3:
        st.button("🔴 At-risk customer example", use_container_width=True,
                  on_click=apply_preset, args=("at_risk",), key="preset_risk_btn")

    with st.form("customer_form"):
        st.subheader("2. Customer details")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
            senior_citizen = st.selectbox(
                "Senior Citizen", ["No", "Yes"], key="senior_citizen",
                help="Is this customer 65 or older? Just Yes or No.",
            )
            partner = st.selectbox(
                "Has Partner", ["No", "Yes"], key="partner",
                help="Does this customer have a spouse or partner? Yes or No.",
            )
            dependents = st.selectbox(
                "Has Dependents", ["No", "Yes"], key="dependents",
                help="Does this customer support children or other dependents? Yes or No.",
            )
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12, key="tenure",
                help="How many months has this customer been with the company? "
                     "Example: a brand-new customer = 1-3, a 5-year customer = 60.",
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0,
                key="monthly_charges",
                help="How much this customer is billed per month, in dollars. "
                     "Typical range in this dataset is about $20-$120.",
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"], key="contract",
                help="What length of contract is the customer on? Month-to-month means "
                     "they can cancel anytime with no penalty - this is the single "
                     "biggest driver of churn risk.",
            )
            paperless_billing = st.selectbox(
                "Paperless Billing", ["No", "Yes"], key="paperless_billing",
                help="Does the customer get bills by email/app instead of paper mail?",
            )
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
                help="How the customer pays their bill each month.",
            )

        with col2:
            phone_service = st.selectbox(
                "Phone Service", ["No", "Yes"], key="phone_service",
                help="Does the customer have home phone service through this company?",
            )
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes"], key="multiple_lines",
                help="Does the customer have more than one phone line?",
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service",
                help="What type of internet, if any. 'No' means they don't have "
                     "internet through this company at all.",
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes"], key="online_security",
                help="Add-on service that protects the customer's internet activity.",
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes"], key="online_backup",
                help="Add-on service that backs up the customer's files online.",
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes"], key="device_protection",
                help="Add-on insurance/warranty for the customer's equipment.",
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes"], key="tech_support",
                help="Does the customer pay for priority technical support?",
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes"], key="streaming_tv",
                help="Does the customer stream TV through this company's internet?",
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes"], key="streaming_movies",
                help="Does the customer stream movies through this company's internet?",
            )

        submitted = st.form_submit_button("🔮 Predict churn risk", use_container_width=True)

    if submitted:
        raw, service_cols_yes = engineer_customer_row(
            gender, senior_citizen, partner, dependents, tenure, phone_service,
            multiple_lines, internet_service, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method, monthly_charges,
        )
        input_df = pd.DataFrame([raw])[meta["X_columns"]]

        proba = float(model.predict_proba(input_df)[0, 1])

        if proba >= 0.66:
            risk_label, risk_color = "High risk of churn", COLOR_CRITICAL
        elif proba >= 0.33:
            risk_label, risk_color = "Moderate risk", COLOR_WARNING
        else:
            risk_label, risk_color = "Likely to stay", COLOR_GOOD

        st.divider()
        st.subheader("3. Result")

        out_of_100 = round(proba * 100)
        if proba >= 0.66:
            st.error(
                f"🚨 **{risk_label}** — out of 100 customers who look like this, "
                f"about **{out_of_100}** are expected to cancel. Consider reaching "
                f"out with a retention offer."
            )
        elif proba >= 0.33:
            st.warning(
                f"⚠️ **{risk_label}** — out of 100 customers who look like this, "
                f"about **{out_of_100}** are expected to cancel. Worth keeping an eye on."
            )
        else:
            st.success(
                f"✅ **{risk_label}** — out of 100 customers who look like this, "
                f"only about **{out_of_100}** are expected to cancel. No action needed."
            )

        left, right = st.columns([1, 1])

        with left:
            st.caption("**Risk meter** — the needle shows the churn probability. "
                       "Green = safe, amber = watch closely, red = act now.")
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": risk_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": GRID_COLOR, "tickfont": {"color": TEXT_COLOR}},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 33], "color": "rgba(12,163,12,0.15)"},
                        {"range": [33, 66], "color": "rgba(237,161,0,0.15)"},
                        {"range": [66, 100], "color": "rgba(208,59,59,0.15)"},
                    ],
                },
                title={"text": "Chance this customer cancels", "font": {"size": 13, "color": TEXT_COLOR}},
            ))
            gauge_fig.update_layout(
                height=260, margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with right:
            st.markdown(f"### :{'red' if risk_color == COLOR_CRITICAL else 'orange' if risk_color == COLOR_WARNING else 'green'}[{risk_label}]")
            st.metric("Predicted churn probability", f"{proba:.1%}")
            st.metric("Customer tenure", f"{tenure} months")
            st.metric("Monthly charges", f"${monthly_charges:,.2f}")

        st.divider()
        st.subheader("4. Why did the model say this?")
        st.caption(
            "The chart below shows the 5 details about this customer that most "
            "affected the prediction. Read it top to bottom - the topmost bar "
            "mattered the most."
        )

        transformed = model.named_steps["prep"].transform(input_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = model.named_steps["prep"].get_feature_names_out()

        explainer = shap.TreeExplainer(model.named_steps["clf"])
        shap_row = explainer(transformed)

        contrib = pd.Series(shap_row.values[0], index=feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:5])
        top = top.iloc[::-1]

        friendly_labels = [
            humanize_feature(f, meta["categorical_cols"], meta["numeric_cols"])
            for f in top.index
        ]
        bar_colors = [COLOR_CRITICAL if v > 0 else COLOR_GOOD for v in top.values]

        shap_fig = go.Figure(go.Bar(
            x=top.values,
            y=friendly_labels,
            orientation="h",
            marker_color=bar_colors,
            text=["Pushes risk UP" if v > 0 else "Pushes risk DOWN" for v in top.values],
            textposition="outside",
        ))
        shap_fig.update_layout(
            height=300,
            margin=dict(l=10, r=120, t=10, b=10),
            xaxis_title="← Makes them less likely to leave   |   Makes them more likely to leave →",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=GRID_COLOR, showticklabels=False),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        st.markdown("**In plain words:**")
        for feature, value, label in zip(top.index[::-1], top.values[::-1], friendly_labels[::-1]):
            if value > 0:
                st.markdown(f"- 🔴 **{label}** — this is *increasing* the customer's risk of leaving.")
            else:
                st.markdown(f"- 🟢 **{label}** — this is *helping keep* the customer (lowering their risk).")

        if "feature_averages" in meta:
            st.divider()
            st.subheader("5. How does this customer compare to a typical one?")
            st.caption(
                "Blue bars = this customer. Gray bars = the average customer "
                "across the whole dataset. This gives you a quick gut-check on "
                "whether this customer looks 'normal' or unusual."
            )

            avgs = meta["feature_averages"]
            compare_fig = go.Figure()
            compare_fig.add_trace(go.Bar(
                name="This customer",
                x=["Tenure (months)", "Monthly Charges ($)"],
                y=[tenure, monthly_charges],
                marker_color=COLOR_BLUE,
            ))
            compare_fig.add_trace(go.Bar(
                name="Average customer",
                x=["Tenure (months)", "Monthly Charges ($)"],
                y=[avgs["tenure"], avgs["MonthlyCharges"]],
                marker_color=COLOR_MUTED,
            ))
            compare_fig.update_layout(
                barmode="group",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_COLOR),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(gridcolor=GRID_COLOR),
            )
            st.plotly_chart(compare_fig, use_container_width=True)

            tenure_note = "shorter than" if tenure < avgs["tenure"] else "longer than"
            charge_note = "more than" if monthly_charges > avgs["MonthlyCharges"] else "less than"
            st.markdown(
                f"**In plain words:** this customer has been here for **{tenure} months**, "
                f"which is **{tenure_note}** the average of **{avgs['tenure']:.0f} months** — "
                f"and pays **${monthly_charges:,.2f}/month**, which is **{charge_note}** "
                f"the average of **${avgs['MonthlyCharges']:,.2f}/month**. Newer, "
                f"higher-paying customers tend to be higher risk."
            )

        st.divider()
        st.subheader("6. What is this customer subscribed to?")
        st.caption(
            "A quick visual summary of this customer's plan. More subscribed "
            "services generally means more reasons to stay (harder to walk "
            "away from an ecosystem of services), but very high total bills "
            "from stacking many add-ons can also push price-sensitive "
            "customers toward leaving - it cuts both ways, which is why this "
            "chart is useful alongside the risk score above, not instead of it."
        )

        service_labels = [
            "Phone Service", "Multiple Lines", "Online Security", "Online Backup",
            "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
        ]
        service_values = service_cols_yes
        n_subscribed = sum(1 for v in service_values if v == "Yes")
        n_total = len(service_values)

        donut_fig = go.Figure(go.Pie(
            labels=["Subscribed", "Not subscribed"],
            values=[n_subscribed, n_total - n_subscribed],
            hole=0.6,
            marker_colors=[COLOR_BLUE, "#3a3a40"],
            textinfo="label+percent",
            textfont=dict(color=TEXT_COLOR),
            sort=False,
        ))
        donut_fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_COLOR),
            annotations=[dict(
                text=f"{n_subscribed}/{n_total}<br>services", x=0.5, y=0.5,
                font=dict(size=16, color=TEXT_COLOR), showarrow=False,
            )],
        )
        st.plotly_chart(donut_fig, use_container_width=True)

        subscribed_list = [lab for lab, val in zip(service_labels, service_values) if val == "Yes"]
        if subscribed_list:
            st.markdown(f"**Services this customer has:** {', '.join(subscribed_list)}.")
        else:
            st.markdown("**Services this customer has:** none of the optional add-ons.")

# =================================================================
# TAB 2: Search Existing Customer
# =================================================================
with tab_search:
    st.subheader("🔍 Search an existing customer")
    st.markdown(
        "Look up any of the **7,043 customers** the models were trained/"
        "demoed on and see their full Health Score - combining churn risk, "
        "usage trend, and support-message sentiment into one number."
    )

    try:
        health_df = load_health_scores()
    except FileNotFoundError:
        health_df = None

    if health_df is None:
        st.warning(
            "⚠️ `customer_health_scores.csv` not found. Run the pipeline "
            "scripts in order first: `generate_synthetic_data.py` → "
            "`train_usage_trend_model.py` → `train_sentiment_model.py` → "
            "`combine_health_score.py`, then restart the app."
        )
    else:
        customer_id = st.selectbox(
            "Select a customer ID", health_df["customerID"].tolist(),
            help="Start typing to filter the list.",
        )
        row = health_df[health_df["customerID"] == customer_id].iloc[0]
        st.divider()
        render_health_breakdown(
            row["churn_risk_score"], row["usage_trend"], row["sentiment_score"],
            row["complaint_category"], row["health_score"], row["risk_tier"], row["explanation"],
        )

# =================================================================
# TAB 3: What-If Simulator
# =================================================================
with tab_whatif:
    st.subheader("🧪 What-If Simulator")
    st.markdown(
        "Type in a hypothetical customer's account details, their last 5 "
        "months of usage, and a sample support message - this tab runs "
        "**all three models live** and shows you the combined Health Score."
    )

    with st.form("whatif_form"):
        st.markdown("**Account details**")
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            wi_contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"], key="wi_contract")
            wi_tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, key="wi_tenure")
            wi_monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0, key="wi_charges")
        with wcol2:
            wi_internet = st.selectbox("Internet service type", ["DSL", "Fiber optic", "No"], key="wi_internet")
            wi_gender = st.selectbox("Gender", ["Female", "Male"], key="wi_gender")
            wi_senior = st.selectbox("Senior citizen", ["No", "Yes"], key="wi_senior")

        st.markdown("**Last 5 months of usage** (minutes/GB - any consistent unit works)")
        ucols = st.columns(5)
        usage_values = []
        for i, uc in enumerate(ucols):
            with uc:
                v = st.number_input(f"Month {i + 1}", min_value=0.0, value=0.0, step=10.0, key=f"wi_usage_{i}")
                usage_values.append(v)

        st.markdown("**Sample support message** (what might this customer say to support?)")
        wi_message = st.text_area(
            "Support message", value="", key="wi_message",
            help="Example: 'My internet keeps dropping and nobody has fixed it.'",
        )

        wi_submit = st.form_submit_button("🧮 Calculate Health Score", use_container_width=True)

    if wi_submit:
        if all(v == 0 for v in usage_values):
            st.warning("⚠️ Please fill in at least some non-zero usage values for all 5 months before calculating.")
        elif not wi_message.strip():
            st.warning("⚠️ Please enter a sample support message before calculating - it's needed for the sentiment score.")
        else:
            raw, _ = engineer_customer_row(
                gender=wi_gender, senior_citizen=wi_senior, partner="No", dependents="No",
                tenure=wi_tenure, phone_service="Yes", multiple_lines="No",
                internet_service=wi_internet, online_security="No", online_backup="No",
                device_protection="No", tech_support="No", streaming_tv="No", streaming_movies="No",
                contract=wi_contract, paperless_billing="Yes", payment_method="Electronic check",
                monthly_charges=wi_monthly_charges,
            )
            input_df = pd.DataFrame([raw])[meta["X_columns"]]
            churn_risk = float(model.predict_proba(input_df)[0, 1])

            usage_model, label_encoder = load_usage_model()
            usage_trend = predict_usage_trend(usage_values, usage_model, label_encoder)

            with st.spinner("Loading sentiment model (first run only takes longer)..."):
                sentiment_pipe = load_sentiment_pipeline()
            if sentiment_pipe is None:
                st.info(
                    "ℹ️ The sentiment model couldn't load in this environment, so the "
                    "support message below was scored as neutral instead. Churn risk "
                    "and usage trend are unaffected."
                )
            sentiment_score = predict_sentiment([wi_message], sentiment_pipe)[0]

            health_score = combine_scores(churn_risk, usage_trend, sentiment_score)
            risk_tier = assign_tier(health_score)
            explanation = generate_explanation({
                "churn_risk_score": churn_risk, "usage_trend": usage_trend, "sentiment_score": sentiment_score,
            })

            st.divider()
            render_health_breakdown(
                churn_risk, usage_trend, sentiment_score, "N/A (single message)",
                health_score, risk_tier, explanation,
            )

# =================================================================
# TAB 4: Batch Upload
# =================================================================
with tab_batch:
    st.subheader("📁 Batch Upload")
    st.markdown(
        "Upload a CSV of many customers to score them all at once. Required "
        "columns: `tenure`, `MonthlyCharges`, `Contract`, `InternetService`. "
        "Optional columns: `month_1_usage` through `month_5_usage` (defaults "
        "to a flat/neutral trend if missing) and `support_message` (defaults "
        "to neutral sentiment if missing)."
    )

    REQUIRED_COLS = ["tenure", "MonthlyCharges", "Contract", "InternetService"]

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"❌ Couldn't read that file as a CSV: {e}")
            batch_df = None

        if batch_df is not None:
            missing_cols = [c for c in REQUIRED_COLS if c not in batch_df.columns]
            if missing_cols:
                st.error(
                    f"❌ This CSV is missing required column(s): **{', '.join(missing_cols)}**. "
                    f"Please add them and re-upload."
                )
            else:
                usage_cols_present = [c for c in batch_df.columns if c.startswith("month_") and c.endswith("_usage")]
                has_usage = len(usage_cols_present) == 5
                has_messages = "support_message" in batch_df.columns

                with st.spinner(f"Scoring {len(batch_df)} customers..."):
                    usage_model, label_encoder = load_usage_model()
                    sentiment_pipe = load_sentiment_pipeline() if has_messages else None

                    churn_risks, usage_trends, sentiment_scores = [], [], []
                    for _, r in batch_df.iterrows():
                        raw, _ = engineer_customer_row(
                            gender=r.get("gender", "Female"),
                            senior_citizen=r.get("SeniorCitizen", "No") if str(r.get("SeniorCitizen", "No")) in ("Yes", "No") else "No",
                            partner=r.get("Partner", "No"), dependents=r.get("Dependents", "No"),
                            tenure=r["tenure"], phone_service=r.get("PhoneService", "Yes"),
                            multiple_lines=r.get("MultipleLines", "No"), internet_service=r["InternetService"],
                            online_security=r.get("OnlineSecurity", "No"), online_backup=r.get("OnlineBackup", "No"),
                            device_protection=r.get("DeviceProtection", "No"), tech_support=r.get("TechSupport", "No"),
                            streaming_tv=r.get("StreamingTV", "No"), streaming_movies=r.get("StreamingMovies", "No"),
                            contract=r["Contract"], paperless_billing=r.get("PaperlessBilling", "Yes"),
                            payment_method=r.get("PaymentMethod", "Electronic check"),
                            monthly_charges=r["MonthlyCharges"],
                        )
                        input_df = pd.DataFrame([raw])[meta["X_columns"]]
                        churn_risks.append(float(model.predict_proba(input_df)[0, 1]))

                        if has_usage:
                            usage_trends.append(predict_usage_trend(r[usage_cols_present].to_numpy(dtype=float), usage_model, label_encoder))
                        else:
                            usage_trends.append("Stable")

                    if has_messages:
                        messages = batch_df["support_message"].fillna("").tolist()
                        messages = [m if m.strip() else "No message provided." for m in messages]
                        sentiment_scores = predict_sentiment(messages, sentiment_pipe)
                    else:
                        sentiment_scores = [0.0] * len(batch_df)

                    results = batch_df.copy()
                    results["churn_risk_score"] = churn_risks
                    results["usage_trend"] = usage_trends
                    results["sentiment_score"] = sentiment_scores
                    results["health_score"] = [
                        combine_scores(cr, ut, ss) for cr, ut, ss in zip(churn_risks, usage_trends, sentiment_scores)
                    ]
                    results["risk_tier"] = results["health_score"].apply(assign_tier)

                st.success(f"✅ Scored {len(results)} customers.")
                if not has_usage:
                    st.caption("ℹ️ No usage history columns found - usage trend defaulted to 'Stable' for all rows.")
                if not has_messages:
                    st.caption("ℹ️ No `support_message` column found - sentiment defaulted to neutral for all rows.")
                elif sentiment_pipe is None:
                    st.caption("ℹ️ The sentiment model couldn't load in this environment - sentiment defaulted to neutral for all rows.")

                tier_filter = st.selectbox("Filter by risk tier", ["All", "Critical", "At-Risk", "Healthy"])
                display_df = results if tier_filter == "All" else results[results["risk_tier"] == tier_filter]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "churn_risk_score": st.column_config.ProgressColumn(
                            "Churn risk", min_value=0, max_value=1, format="%.0%%"
                        ),
                        "health_score": st.column_config.ProgressColumn(
                            "Health score", min_value=0, max_value=100, format="%.0f"
                        ),
                    },
                )

                st.download_button(
                    "⬇️ Download results as CSV",
                    data=display_df.to_csv(index=False).encode("utf-8"),
                    file_name="scored_customers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
