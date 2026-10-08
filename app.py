import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Diagnostic & Risk Prediction")
st.write(
    "Clinical decision support system evaluating cardiovascular disease probability "
    "using an optimized Random Forest Classifier."
)

# ---------------------------------------------------------
# Load Saved Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/heart_disease_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------------------------------------
# Clinical Input Form
# ---------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient Clinical Parameters")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=120, value=54, step=1)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "False (<= 120)" if x == 0 else "True (> 120)"
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
        )
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4], index=0)
        thal = st.selectbox(
            "Thalassemia Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Null/Unknown",
                1: "Fixed Defect",
                2: "Normal",
                3: "Reversible Defect"
            }[x]
        )

    submitted = st.form_submit_button("Analyze Cardiac Risk")

# ---------------------------------------------------------
# Inference & Diagnostics
# ---------------------------------------------------------
if submitted:
    # Feature order exactly matches training dataset
    feature_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([feature_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Diagnostic Assessment")

    if prediction == 1:
        st.error("⚠️ **High Risk:** Indicators suggest high probability of Heart Disease.")
    else:
        st.success("✅ **Low Risk:** Indicators suggest low probability of Heart Disease.")

    st.metric(label="Calculated Disease Probability", value=f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))

st.caption("Model: Random Forest Classifier (Accuracy: ~98.5%) | Dataset: UCI/Kaggle Heart Disease")