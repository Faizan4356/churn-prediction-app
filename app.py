"""
Streamlit app - Customer Churn Prediction

Run locally with:
    streamlit run app.py

Requires churn_model.joblib and churn_model_meta.joblib to exist in the
same directory (produced by train_and_save_model.py).
"""

import pandas as pd
import streamlit as st
import joblib
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------
# Color palette - status colors (good/warning/critical) for churn
# risk, a fixed categorical pair for comparison charts. Kept as
# constants so every chart and card uses the same meaning-to-color
# mapping instead of picking colors ad hoc.
# ---------------------------------------------------------------
COLOR_GOOD = "#0ca30c"       # low risk / retained
COLOR_WARNING = "#eda100"    # medium risk
COLOR_CRITICAL = "#d03b3b"   # high risk / churned
COLOR_BLUE = "#2a78d6"       # primary / neutral series
COLOR_MUTED = "#898781"      # secondary / "average customer" comparisons
GRID_COLOR = "#e1e0d9"

# ---------------------------------------------------------------
# Plain-language labels used to translate raw column/feature names
# into sentences a non-technical reader can follow. This is the
# single place that maps "MonthlyCharges" -> "how much they pay per
# month" etc, so both the chart labels and the explanation sentences
# stay consistent.
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
        background: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }}
    /* Soft colorful page background - a light blue-to-lavender wash
       behind everything, with white "cards" for the form and the
       main block so text stays readable on top of it. */
    [data-testid="stAppViewContainer"] > .main {{
        background: linear-gradient(180deg, #eaf1fc 0%, #f3eefc 45%, #fdf6f0 100%);
    }}
    div[data-testid="stForm"] {{
        background: rgba(255,255,255,0.75);
        border-radius: 14px;
        padding: 1.25rem 1.5rem 0.5rem 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
    }}
    div.block-container div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.6);
        border-radius: 12px;
    }}
    </style>
    <div class="hero-banner">
        <h1>📉 Customer Churn Predictor</h1>
        <p>Fill in a customer's details (or click an example below) to see how likely
        they are to cancel their service, explained in plain English.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("❓ New here? Read this first (30 seconds)", expanded=False):
    st.markdown(
        """
        **What this app does:** it looks at one customer's details — how long
        they've been a customer, what contract they're on, how much they
        pay, which add-on services they use — and predicts how likely they
        are to **cancel their service** ("churn"). It's built on a machine
        learning model (XGBoost) trained on 7,043 real telecom customers,
        about a quarter of whom actually churned.

        **Why this matters for a business:** it's far cheaper to keep an
        existing customer than to win a new one. If a company can flag
        "this customer is 90% likely to leave" *before* they leave, they
        can step in with a discount, a phone call, or a better plan —
        instead of finding out only after the customer has already
        cancelled.

        **You don't need to know what to type.** Click one of the three
        example buttons below the form title — *Loyal customer*,
        *Average customer*, or *At-risk customer* — to instantly fill the
        form with a realistic example, then hit **Predict**. Once you see
        how it works, feel free to change any field yourself (like tenure
        or contract type) and predict again to see how the result changes.

        **What you'll get back, in order:**
        1. A **risk score** (0-100%) — how likely this customer is to leave,
           shown as a plain sentence and a color-coded speedometer.
        2. The **top reasons** behind that score, translated into plain
           sentences (not technical jargon).
        3. How this customer **compares to a typical customer** in the data.
        4. A breakdown of which **services** this customer is subscribed to.
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

model, meta = load_model()

# ---------------------------------------------------------------
# 0a. Dataset-wide reference charts - these are NOT about any one
# customer. They show patterns across all 7,043 customers the model
# was trained on, so a reader has context for *why* the model cares
# about contract type and tenure before they even predict anything.
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 0. Quick-fill example buttons (outside the form so a click can
#    refill the form's fields before the user submits anything)
# ---------------------------------------------------------------
st.subheader("1. Pick an example, or fill in your own")
b1, b2, b3 = st.columns(3)
with b1:
    st.button("🟢 Loyal customer example", use_container_width=True,
              on_click=apply_preset, args=("loyal",))
with b2:
    st.button("🟡 Average customer example", use_container_width=True,
              on_click=apply_preset, args=("average",))
with b3:
    st.button("🔴 At-risk customer example", use_container_width=True,
              on_click=apply_preset, args=("at_risk",))

# ---------------------------------------------------------------
# 1. Input form - every field has a plain-language "help" tooltip
#    (hover the small ? icon) explaining what to put and why it
#    matters, since most fields aren't self-explanatory to someone
#    who hasn't seen a telecom billing form before.
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 2. Build feature row + predict
# ---------------------------------------------------------------
if submitted:
    total_charges = tenure * monthly_charges  # best estimate without real billing history

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

    # Recreate the same engineered features from Phase 4
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

    # Plain-language headline box before any chart, so the takeaway is
    # clear even if someone skips the charts entirely.
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
        # Gauge chart with three color zones (good/warning/critical) so
        # the risk level reads instantly, not just from the number.
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 40, "color": risk_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": GRID_COLOR},
                "bar": {"color": risk_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(12,163,12,0.15)"},
                    {"range": [33, 66], "color": "rgba(237,161,0,0.15)"},
                    {"range": [66, 100], "color": "rgba(208,59,59,0.15)"},
                ],
            },
            title={"text": "Chance this customer cancels", "font": {"size": 13}},
        ))
        gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with right:
        st.markdown(f"### :{'red' if risk_color == COLOR_CRITICAL else 'orange' if risk_color == COLOR_WARNING else 'green'}[{risk_label}]")
        st.metric("Predicted churn probability", f"{proba:.1%}")
        st.metric("Customer tenure", f"{tenure} months")
        st.metric("Monthly charges", f"${monthly_charges:,.2f}")

    # -----------------------------------------------------------
    # 3. Explain the prediction with SHAP - shown as both a chart
    #    AND plain-English sentences, since a bar chart of encoded
    #    feature names ("cat__Contract_Month-to-month") means nothing
    #    without translation.
    # -----------------------------------------------------------
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
    # Reverse so the largest contributor plots at the top of the bar chart
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
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=GRID_COLOR, showticklabels=False),
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    st.markdown("**In plain words:**")
    for feature, value, label in zip(top.index[::-1], top.values[::-1], friendly_labels[::-1]):
        if value > 0:
            st.markdown(f"- 🔴 **{label}** — this is *increasing* the customer's risk of leaving.")
        else:
            st.markdown(f"- 🟢 **{label}** — this is *helping keep* the customer (lowering their risk).")

    # -----------------------------------------------------------
    # 4. Customer vs. typical customer - quick visual context, with
    #    a plain-language takeaway sentence under the chart.
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 5. Which services this customer has - a donut chart. This is
    # the most approachable chart in the app (just a subscribed vs.
    # not-subscribed split), useful as a plain visual summary of the
    # customer's plan before/after reading the risk analysis above.
    # -----------------------------------------------------------
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
    service_values = service_cols_yes  # already collected above, same order
    n_subscribed = sum(1 for v in service_values if v == "Yes")
    n_total = len(service_values)

    donut_fig = go.Figure(go.Pie(
        labels=["Subscribed", "Not subscribed"],
        values=[n_subscribed, n_total - n_subscribed],
        hole=0.6,
        marker_colors=[COLOR_BLUE, "#dcdcdc"],
        textinfo="label+percent",
        sort=False,
    ))
    donut_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        annotations=[dict(
            text=f"{n_subscribed}/{n_total}<br>services", x=0.5, y=0.5,
            font_size=16, showarrow=False,
        )],
    )
    st.plotly_chart(donut_fig, use_container_width=True)

    subscribed_list = [lab for lab, val in zip(service_labels, service_values) if val == "Yes"]
    if subscribed_list:
        st.markdown(f"**Services this customer has:** {', '.join(subscribed_list)}.")
    else:
        st.markdown("**Services this customer has:** none of the optional add-ons.")
