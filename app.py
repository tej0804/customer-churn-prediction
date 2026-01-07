import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model & scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction Dashboard")

st.sidebar.header("Customer Behavior Inputs")

frequency = st.sidebar.number_input("Purchase Frequency", min_value=1, value=5)
monetary = st.sidebar.number_input("Total Spend", min_value=1.0, value=500.0)
avg_order_value = st.sidebar.number_input("Avg Order Value", min_value=1.0, value=100.0)
purchase_intensity = st.sidebar.number_input("Purchase Intensity", min_value=0.01, value=0.05)

input_data = np.array([[frequency, monetary, avg_order_value, purchase_intensity]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.metric("Churn Probability", f"{prob:.2f}")

    if prob > 0.6:
        st.error("High Churn Risk 🚨")
    elif prob > 0.3:
        st.warning("Medium Churn Risk ⚠️")
    else:
        st.success("Low Churn Risk ✅")
