"""
E-Commerce Return Fraud Detection - Streamlit App

"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Return Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------
# CUSTOM CSS (good-looking theme)
# ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1 { color: #4F8BF9; }
    .risk-badge {
        padding: 14px; border-radius: 10px; text-align: center;
        font-size: 20px; font-weight: 700; color: white; margin: 10px 0;
    }
    .factor-row {
        padding: 8px 12px; border-radius: 8px; margin: 5px 0;
        background: #1c1f26; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL + SCALER + COLUMNS (cached)

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(NOTEBOOKS_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(NOTEBOOKS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(NOTEBOOKS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, scaler, feature_columns = load_artifacts()
explainer = load_explainer(model)



# RISK CATEGORY FUNCTION

def risk_category(prob):
    if prob < 0.30:
        return ("Low Risk", "🟢", "#1D9E75",
                "✅ Auto-approve the return. Process refund normally. "
                "Genuine customer — avoid unnecessary friction.")
    elif prob < 0.70:
        return ("Medium Risk", "🟡", "#E2A03F",
                "⚠️ Manual review recommended. Verify order history "
                "and return reason before approving.")
    else:
        return ("High Risk", "🔴", "#E24B4A",
                "🔴 Hold refund & investigate. Do NOT auto-refund. "
                "Flag for fraud team review.")



# TITLE

st.title("🛡️ E-Commerce Return Fraud Detection")
st.markdown("Enter order & customer details to assess **fraud risk**")
st.divider()


# LAYOUT: left = input, right = result

col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("📝 Customer & Order Details")

    c1, c2 = st.columns(2)
    customer_age = c1.number_input("Customer Age", 18, 80, 35)
    account_age_days = c2.number_input("Account Age (days)", 1, 3650, 365)

    c3, c4 = st.columns(2)
    total_orders = c3.number_input("Total Orders", 1, 500, 50)
    previous_returns = c4.number_input("Previous Returns", 0, 200, 10)

    c5, c6 = st.columns(2)
    ip_risk_score = c5.slider("IP Risk Score", 0, 100, 40)
    product_price = c6.number_input("Product Price (₹)", 0, 200000, 5000)

    c7, c8 = st.columns(2)
    customer_support_tickets = c7.number_input("Support Tickets", 0, 50, 2)
    days_to_return = c8.number_input("Days to Return", 0, 60, 10)

    product_category = st.selectbox("Product Category",
        ["Books", "Electronics", "Fashion", "Furniture", "Mobile"])
    payment_method = st.selectbox("Payment Method",
        ["Credit Card", "Debit Card", "Net Banking", "UPI"])

    c9, c10 = st.columns(2)
    membership_type = c9.selectbox("Membership", ["Gold", "Silver", "Platinum"])
    city_tier = c10.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    predict_btn = st.button("🔍 Predict Fraud Risk", type="primary",
                            use_container_width=True)


with col_result:
    st.subheader("📊 Fraud Risk Assessment")

    if predict_btn:
        # ---- engineered features ----
        return_rate = previous_returns / total_orders if total_orders > 0 else 0
        account_age_years = account_age_days / 365
        is_new_account = 1 if account_age_days < 90 else 0
        high_return_rate = 1 if return_rate > 0.25 else 0
        support_intensity = customer_support_tickets / total_orders if total_orders > 0 else 0
        high_ip_risk = 1 if ip_risk_score > 56 else 0
        fast_return = 1 if days_to_return < 3 else 0

        membership_map = {"Gold": 1, "Silver": 2, "Platinum": 3}
        city_map = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}

        # all 41 features (defaults for un-asked) 
        input_data = {c: 0 for c in feature_columns}
        input_data.update({
            'customer_age': customer_age, 'gender': 1,
            'city_tier': city_map[city_tier],
            'membership_type': membership_map[membership_type],
            'account_age_days': account_age_days, 'total_orders': total_orders,
            'previous_returns': previous_returns, 'return_rate': return_rate,
            'product_price': product_price, 'discount_percent': 20,
            'order_amount': product_price, 'delivery_days': 5,
            'days_to_return': days_to_return, 'customer_rating': 3,
            'customer_support_tickets': customer_support_tickets,
            'ip_risk_score': ip_risk_score,
            'return_reason_Quality Issue': 1, 'device_type_Desktop': 1,
            'order_month': 6, 'order_dayofweek': 3, 'is_weekend_order': 0,
            'account_age_years': account_age_years,
            'is_new_account': is_new_account, 'high_return_rate': high_return_rate,
            'support_intensity': support_intensity, 'high_ip_risk': high_ip_risk,
            'fast_return': fast_return,
        })
        input_data[f'product_category_{product_category}'] = 1
        input_data[f'payment_method_{payment_method}'] = 1

        input_df = pd.DataFrame([input_data])[feature_columns]

        #predict
        fraud_proba = model.predict_proba(input_df)[0][1]
        category, emoji, color, recommendation = risk_category(fraud_proba)

        # display 
        st.metric("Fraud Probability", f"{fraud_proba:.1%}")
        st.progress(float(fraud_proba))
        st.markdown(
            f"<div class='risk-badge' style='background:{color};'>{emoji} {category}</div>",
            unsafe_allow_html=True)

        st.markdown("#### 💡 Recommendation")
        st.info(recommendation)

        # SHAP key factors 
        st.markdown("#### 🔑 Key Risk Factors")
        shap_values = explainer.shap_values(input_df)
        shap_series = pd.Series(shap_values[0], index=feature_columns)
        top_factors = shap_series.abs().sort_values(ascending=False).head()

        for feat in top_factors.index:
            impact = shap_series[feat]
            value = input_df[feat].values[0]
            if impact > 0:
                st.markdown(
                    f"<div class='factor-row' style='color:#ADD8E6;'>🔴 ⬆️ <b>{feat}</b> = {value:.2f} "
                    f"<i>(increases fraud risk)</i></div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='factor-row' style='color:#ADD8E6;'>🟢 ⬇️ <b>{feat}</b> = {value:.2f} "
                    f"<i>(reduces fraud risk)</i></div>", unsafe_allow_html=True)
    else:
        st.info("👈 Enter details and click 'Predict Fraud Risk' to see results")

        ## Runcode streamlit run app.py