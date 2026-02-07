import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# App title
st.title("Diabetes Prediction App")

st.write("Enter patient details to predict diabetes status")

# User inputs
glucose = st.number_input("Glucose Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[glucose, bmi, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")
