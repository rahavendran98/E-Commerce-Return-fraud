import streamlit as st
import pandas as pd
import joblib
import numpy as np

#page_configuration

st.set_page_config(page_title="Return Fraud Detection", page_icon="🛡️", layout="wide")

#Loading model and scaler and feature columns

@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("feature_columns.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

#Title
st.title("🛡️ E-Commerce Return Fraud Detection")
st.markdown("Enter order & custoemr details to assess fraud risk")

#Two columns Left = Input Right = Result

# Title
st.title("🛡️ E-Commerce Return Fraud Detection")
st.markdown("Enter order & customer details to assess fraud risk")

# Two columns: left = input, right = result
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("📝 Customer & Order Details")

    # Row 1: customer basics
    c1, c2 = st.columns(2)
    customer_age = c1.number_input("Customer Age", 18, 80, 35)
    account_age_days = c2.number_input("Account Age (days)", 1, 3650, 365)

    # Row 2: order behaviour 
    c3, c4 = st.columns(2)
    total_orders = c3.number_input("Total Orders", 1, 500, 50)
    previous_returns = c4.number_input("Previous Returns", 0, 200, 10)

    # Row 3: risk + money
    c5, c6 = st.columns(2)
    ip_risk_score = c5.slider("IP Risk Score", 0, 100, 40)
    product_price = c6.number_input("Product Price (₹)", 0, 200000, 5000)

    # Row 4: more details
    c7, c8 = st.columns(2)
    customer_support_tickets = c7.number_input("Support Tickets", 0, 50, 2)
    days_to_return = c8.number_input("Days to Return", 0, 60, 10)

    # Row 5: categorical dropdowns
    product_category = st.selectbox("Product Category",
        ["Books", "Electronics", "Fashion", "Furniture", "Mobile"])
    payment_method = st.selectbox("Payment Method",
        ["Credit Card", "Debit Card", "Net Banking", "UPI"])

    # Row 6: more categorical
    membership_type = st.selectbox("Membership", ["Gold", "Silver", "Platinum"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    # Predict button
    predict_btn = st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True)    