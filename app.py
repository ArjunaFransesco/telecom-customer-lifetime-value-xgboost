import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telecom Customer Lifetime Value (CLV) & Margin Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Telecom Customer Lifetime Value (CLV) & Margin Predictor")
st.markdown("**Domain**: `Telecommunications / Marketing ML` | **Tech Stack**: `XGBoost, Scikit-Learn, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    monthly_tenure_months = st.slider("Monthly Tenure Months", float(1.0), float(72.0), float(28.0))
    monthly_charge_usd = st.slider("Monthly Charge Usd", float(18.0), float(140.0), float(68.0))
    support_tickets_count = st.slider("Support Tickets Count", int(0), int(10), int(2))
    contract_commitment_tier = st.slider("Contract Commitment Tier", int(1), int(3), int(1))
    service_usage_gigabytes = st.slider("Service Usage Gigabytes", float(10.0), float(500.0), float(145.0))
    satisfaction_nps_score = st.slider("Satisfaction Nps Score", float(1.0), float(10.0), float(7.2))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"monthly_tenure_months": monthly_tenure_months, "monthly_charge_usd": monthly_charge_usd, "support_tickets_count": support_tickets_count, "contract_commitment_tier": contract_commitment_tier, "service_usage_gigabytes": service_usage_gigabytes, "satisfaction_nps_score": satisfaction_nps_score}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `is_attrition_risk`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
