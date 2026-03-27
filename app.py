# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load model and columns
with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)
columns = pickle.load(open("columns.pkl", "rb"))

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        💳 AI Powered Fraud Detection System
    </h1>
    <p style='text-align: center;'>
        Enter transaction details to predict whether it is Fraud or Legitimate.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("📥 Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Origin)", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Destination)", min_value=0.0)

type_option = st.sidebar.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

hours = st.sidebar.slider(
    "Hours to Complete Transaction",
    min_value=1,
    max_value=800
)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Check Fraud")

# ---------- MAIN CONTENT ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Transaction Summary")
    st.write(f"**Amount:** ₹ {amount}")
    st.write(f"**Type:** {type_option}")
    st.write(f"**Completion Time:** {hours} hours")

with col2:
    st.subheader("💼 Balance Information")
    st.write(f"Origin: {oldbalanceOrg} ➜ {newbalanceOrig}")
    st.write(f"Destination: {oldbalanceDest} ➜ {newbalanceDest}")

st.markdown("---")

# ---------- PREDICTION ----------
if predict_button:

    input_dict = {col: 0 for col in columns}

    input_dict['amount'] = amount
    input_dict['oldbalanceOrg'] = oldbalanceOrg
    input_dict['newbalanceOrig'] = newbalanceOrig
    input_dict['oldbalanceDest'] = oldbalanceDest
    input_dict['newbalanceDest'] = newbalanceDest
    input_dict['hours'] = hours

    type_column = f"type_{type_option}"
    if type_column in input_dict:
        input_dict[type_column] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Prediction Result")

    # Progress bar
    st.progress(int(probability * 100))

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
        st.markdown(
            f"### Fraud Probability: **{probability:.2%}**"
        )
    else:
        st.success("✅ Legitimate Transaction")
        st.markdown(
            f"### Fraud Probability: **{probability:.2%}**"
        )

    st.markdown("---")

    # Risk Level Indicator
    if probability < 0.3:
        st.info("🟢 Risk Level: LOW")
    elif probability < 0.7:
        st.warning("🟡 Risk Level: MEDIUM")
    else:
        st.error("🔴 Risk Level: HIGH")