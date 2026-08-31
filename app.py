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
    </style>
    <div class="hero-banner">
        <h1>📉 Customer Churn Predictor</h1>
        <p>Enter a customer's details to estimate churn risk and see what's driving it.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
# 1. Input form
# ---------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

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
    st.subheader("Prediction")

    left, right = st.columns([1, 1])

    with left:
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
            title={"text": "Churn Probability", "font": {"size": 14}},
        ))
        gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with right:
        st.markdown(f"### :{'red' if risk_color == COLOR_CRITICAL else 'orange' if risk_color == COLOR_WARNING else 'green'}[{risk_label}]")
        st.metric("Predicted churn probability", f"{proba:.1%}")
        st.metric("Customer tenure", f"{tenure} months")
        st.metric("Monthly charges", f"${monthly_charges:,.2f}")

    # -----------------------------------------------------------
    # 3. Explain the prediction with SHAP - top factors, as a chart
    # -----------------------------------------------------------
    st.divider()
    st.subheader("Why this prediction?")

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

    clean_names = [f.split("__", 1)[-1].replace("_", " ", 1) for f in top.index]
    bar_colors = [COLOR_CRITICAL if v > 0 else COLOR_GOOD for v in top.values]

    shap_fig = go.Figure(go.Bar(
        x=top.values,
        y=clean_names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in top.values],
        textposition="outside",
    ))
    shap_fig.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Impact on churn probability",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=GRID_COLOR),
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    st.caption(
        f"🔴 Red bars increase churn risk · 🟢 Green bars decrease it. "
        "Bar length shows how much that feature moved this prediction."
    )

    # -----------------------------------------------------------
    # 4. Customer vs. typical customer - quick visual context
    # -----------------------------------------------------------
    if "feature_averages" in meta:
        st.divider()
        st.subheader("This customer vs. a typical customer")

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
