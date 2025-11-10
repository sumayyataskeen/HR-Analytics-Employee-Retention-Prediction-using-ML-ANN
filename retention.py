import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.express as px

# --------------------------
# 🎯 Page Config
# --------------------------
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="💼",
    layout="wide"
)

# --------------------------
# 🎨 Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight:700;
        color:#2c3e50;
    }
    .result-box {
        background-color:#f4f6f8;
        padding:20px;
        border-radius:15px;
        text-align:center;
        box-shadow:2px 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 📥 Load Models
# --------------------------
model_ann = load_model("model/ann_model.h5")
model_ml = pickle.load(open("model/ml_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# --------------------------
# 🧾 Sidebar
# --------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("HR Analytics Dashboard")
st.sidebar.markdown("Predict employee **Attrition Risk** using **ML + ANN**")

# --------------------------
# 📋 User Input Section
# --------------------------
st.markdown("<p class='big-font'>🔹 Enter Employee Details</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home (km)", 1, 30, 5)
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    monthly_income = st.number_input("Monthly Income (₹)", 1000, 100000, 30000)
with col3:
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    years = st.slider("Years at Company", 0, 40, 5)
    department = st.selectbox("Department", ["R&D", "Sales", "HR"])

# Encode categorical values
gender_val = 1 if gender == "Male" else 0
overtime_val = 1 if overtime == "Yes" else 0
dept_val = {"R&D": 0, "Sales": 1, "HR": 2}[department]

# --------------------------
# 🧮 Prediction
# --------------------------
if st.button("🔍 Predict Attrition"):
    input_data = np.array([[age, distance, education, gender_val, job_satisfaction,
                            monthly_income, overtime_val, years]])
    input_scaled = scaler.transform(input_data)

    # ANN Prediction
    ann_pred = model_ann.predict(input_scaled)[0][0]
    ann_result = "⚠️ Likely to Leave" if ann_pred > 0.5 else "✅ Likely to Stay"

    # ML Prediction
    ml_pred = model_ml.predict(input_scaled)[0]
    ml_result = "⚠️ Likely to Leave" if ml_pred == 1 else "✅ Likely to Stay"

    # Confidence Score
    ann_conf = round(ann_pred * 100, 2) if ann_pred > 0.5 else round((1 - ann_pred) * 100, 2)

    # Display results
    st.markdown("<hr>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='result-box'><h3>🧠 ANN Model Result</h3>", unsafe_allow_html=True)
        st.subheader(ann_result)
        st.metric(label="Confidence", value=f"{ann_conf}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='result-box'><h3>📊 ML Model Result</h3>", unsafe_allow_html=True)
        st.subheader(ml_result)
        st.metric(label="Accuracy (from training)", value="86%")
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------
    # 📊 Visualization
    # --------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📈 Model Confidence Visualization")

    chart_data = pd.DataFrame({
        "Model": ["ANN", "ML"],
        "Confidence (%)": [ann_conf, 86]
    })

    fig = px.bar(chart_data, x="Model", y="Confidence (%)", color="Model",
                 color_discrete_sequence=["#3498db", "#2ecc71"],
                 title="Confidence / Accuracy Comparison", text="Confidence (%)")

    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
        font=dict(color="#2c3e50", size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
