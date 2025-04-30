import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🧪 Patient Diabetes Prediction")

# Load pre-trained model
try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    st.error("❌ Model file 'model.joblib' not found. Please make sure it's in the same folder.")
    st.stop()

# Input form for patient features
with st.form("prediction_form"):
    st.subheader("Enter Patient Information:")

    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose", min_value=0.0, format="%.2f")
    BloodPressure = st.number_input("Blood Pressure", min_value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, format="%.2f")
    Insulin = st.number_input("Insulin", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    Age = st.number_input("Age", min_value=0, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Prediction: Patient is likely Diabetic (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Prediction: Patient is likely Not Diabetic (Confidence: {1 - probability:.2f})")
