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

if st.button("Predict Churn"):
    X = np.array([[frequency, monetary, avg_order_value, purchase_intensity]])
    X_scaled = scaler.transform(X)

    # SAFE for both LR and XGBoost
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        prob = model.predict(X_scaled)[0]

    st.write(f"Churn Probability: **{prob:.2f}**")