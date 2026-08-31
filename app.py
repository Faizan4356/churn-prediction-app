"""
Streamlit app - Customer Churn Prediction

Run locally with:
    streamlit run app.py

Requires churn_model.joblib and churn_model_meta.joblib to exist in the
same directory (produced by train_and_save_model.py).
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------
# Load model (cached so it's loaded once per session, not per click)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.joblib")
    meta = joblib.load("churn_model_meta.joblib")
    return model, meta

model, meta = load_model()

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to estimate their probability of "
    "churning and see the top factors driving that prediction."
)

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

    submitted = st.form_submit_button("Predict")

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

    proba = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction")
    risk_label = "High risk of churn" if proba >= 0.5 else "Likely to stay"
    color = "red" if proba >= 0.5 else "green"
    st.markdown(f"### :{color}[{risk_label}]")
    st.metric("Predicted churn probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    # -----------------------------------------------------------
    # 3. Explain the prediction with SHAP - top 3 factors
    # -----------------------------------------------------------
    st.subheader("Why this prediction?")

    transformed = model.named_steps["prep"].transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = model.named_steps["prep"].get_feature_names_out()

    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_row = explainer(transformed)

    contrib = pd.Series(shap_row.values[0], index=feature_names)
    top3 = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:3])

    for feature, value in top3.items():
        # Clean up encoded names like "cat__Contract_Month-to-month" -> "Contract: Month-to-month"
        clean_name = feature.split("__", 1)[-1].replace("_", " ", 1)
        direction = "increased" if value > 0 else "decreased"
        st.write(f"- **{clean_name}** {direction} churn risk (impact: {value:+.3f})")

    st.caption(
        "Positive impact pushes the prediction toward 'will churn'; "
        "negative impact pushes it toward 'will stay'."
    )
