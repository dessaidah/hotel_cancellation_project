import streamlit as st
import pandas as pd
import joblib
import gdown
import os

model_path = "rf_model_streamlit.pkl"

@st.cache_resource
def load_model():

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1GT3EX9hdRExacBk4Up--Eg128OYEGxgB"
        gdown.download(url, model_path, quiet=False)

    model = joblib.load(model_path)
    return model

model = load_model()

# Load hotel list
hotel_list = joblib.load("hotel_list.pkl")

# Page configuration
st.set_page_config(
    page_title="Hotel Cancellation Risk",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Risk Scoring")

st.write(
"""
This tool predicts the probability that a hotel booking will be cancelled.

The *risk score ranges from 0% to 100%*.
"""
)

# Sidebar input
st.sidebar.header("Booking Information")

hotel = st.sidebar.selectbox(
    "Hotel",
    hotel_list
)

lead_time = st.sidebar.slider(
    "Lead Time (days before arrival)",
    0, 365, 30
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    ["No Deposit", "Refundable", "Non Refund"]
)

market_segment = st.sidebar.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)

adr = st.sidebar.number_input(
    "Average Daily Rate",
    min_value=0.0,
    value=100.0
)

special_requests = st.sidebar.slider(
    "Total Special Requests",
    0,5,0
)

previous_cancel = st.sidebar.selectbox(
    "Previous Cancellations",
    ["0","1+"]
)

# Input dataframe
input_data = pd.DataFrame({
    "hotel":[hotel],
    "lead_time":[lead_time],
    "deposit_type":[deposit_type],
    "market_segment":[market_segment],
    "adr":[adr],
    "total_of_special_requests":[special_requests],
    "previous_cancel_group":[previous_cancel]
})

# Prediction
if st.button("Predict Cancellation Risk"):

    probability = model.predict_proba(input_data)[0][1]
    risk_score = probability * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score")

        st.metric(
            label="Cancellation Probability",
            value=f"{risk_score:.2f}%"
        )

        st.progress(int(risk_score))

    with col2:
        st.subheader("Risk Category")

        if risk_score > 70:
            st.error("High Risk of Cancellation")
        elif risk_score > 40:
            st.warning("Medium Risk of Cancellation")
        else:
            st.success("Low Risk of Cancellation")

st.markdown("---")
st.write("Model: Random Forest | Feature set optimized for prediction app")
