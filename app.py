from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).parent

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered",
)

# ─── Load model & preprocessor once ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = tf.keras.models.load_model(BASE_DIR / "model" / "model.keras")
    preprocessor = joblib.load(BASE_DIR / "model" / "preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see whether they are likely to leave.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    credit_score   = st.number_input("Credit Score",      min_value=300, max_value=900, value=650, step=1)
    geography      = st.selectbox("Geography",            ["France", "Germany", "Spain"])
    gender         = st.selectbox("Gender",               ["Male", "Female"])
    age            = st.slider("Age",                     min_value=18,  max_value=92,  value=38)
    tenure         = st.slider("Tenure (years with bank)", min_value=0, max_value=10,  value=5)

with col2:
    balance        = st.number_input("Account Balance (€)", min_value=0.0,     max_value=300000.0, value=60000.0, step=100.0)
    num_products   = st.selectbox("Number of Products",    [1, 2, 3, 4])
    has_cr_card    = st.selectbox("Has Credit Card?",      ["Yes", "No"])
    is_active      = st.selectbox("Active Member?",        ["Yes", "No"])
    est_salary     = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=250000.0, value=50000.0, step=500.0)

st.divider()

# ─── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):

    # Build one-row DataFrame with exact column names from training
    input_data = pd.DataFrame([{
        "CreditScore":      credit_score,
        "Geography":        geography,
        "Gender":           gender,
        "Age":              age,
        "Tenure":           tenure,
        "Balance":          balance,
        "NumOfProducts":    int(num_products),
        "HasCrCard":        1 if has_cr_card == "Yes" else 0,
        "IsActiveMember":   1 if is_active  == "Yes" else 0,
        "EstimatedSalary":  est_salary,
    }])

    # Transform using the saved preprocessor
    input_proc = preprocessor.transform(input_data)

    # Predict
    prob = float(model.predict(input_proc, verbose=0)[0][0])
    pct  = round(prob * 100, 1)

    st.subheader("Prediction Result")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Churn Probability", f"{pct}%")
    with col_b:
        if prob >= 0.5:
            st.error("⚠️ This customer is **likely to LEAVE** the bank.")
        else:
            st.success("✅ This customer is **likely to STAY** with the bank.")

    # Probability bar
    st.progress(prob)
    st.caption("0% = definitely stays · 100% = definitely leaves")
