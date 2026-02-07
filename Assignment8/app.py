import streamlit as st
import numpy as np
import os
import joblib

# =========================
# Load Model & Scaler Safely
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "logistic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error("❌ Error loading model or scaler files")
    st.write(e)
    st.stop()

# =========================
# Streamlit Page Settings
# =========================
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient health details to predict diabetes")

# =========================
# User Input Fields
# =========================
pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 33)

# =========================
# Prediction
# =========================
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High risk of Diabetes")
    else:
        st.success("✅ Low risk of Diabetes")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
