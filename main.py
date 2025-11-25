# -----------------------------
# Diabetes Prediction App (Streamlit)
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes risk using a trained ML model.")

# -----------------------------
# Load trained model, scaler, and columns
# -----------------------------
try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Make sure diabetes_model.pkl, scaler.pkl, and model_columns.pkl are in the folder.")
    st.stop()

# -----------------------------
# User Input
# -----------------------------
st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hbA1c = st.number_input("HbA1c Level", min_value=3.5, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

# Gender one-hot
gender = st.selectbox("Gender", ["Male", "Female"])
gender_Female = 1 if gender == "Female" else 0
gender_Male = 1 if gender == "Male" else 0

# Race one-hot
race = st.selectbox("Race", ["AfricanAmerican","Asian","Caucasian","Hispanic","Other"])
race_dict = {f"race:{r}": int(r==race) for r in ["AfricanAmerican","Asian","Caucasian","Hispanic","Other"]}

# Smoking one-hot
smoking = st.selectbox("Smoking History", ["Current", "Ever", "Former", "Never", "Not current", "No Info"])
smoking_dict = {f"smoking_history_{s.lower().replace(' ', '_')}": int(s==smoking) 
                for s in ["Current","Ever","Former","Never","Not current","No Info"]}

# Fixed features
year = 2020
hypertension = 0

# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_dict = {
    "year": year,
    "age": age,
    "gender_Female": gender_Female,
    "gender_Male": gender_Male,
    "hypertension": hypertension,
    "bmi": bmi,
    "hbA1c_level": hbA1c,
    "blood_glucose_level": blood_glucose
}
input_dict.update(race_dict)
input_dict.update(smoking_dict)

input_df = pd.DataFrame([input_dict])

# Ensure all columns exist in correct order
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# Scale input
input_scaled = scaler.transform(input_df)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(input_scaled)[0][1] * 100
    else:
        risk_score = None

    st.subheader("📊 Prediction Result")
    st.write(f"**Diabetes Prediction:** {prediction}")
    
    if risk_score is not None:
        st.write(f"**Risk Score:** {risk_score:.2f}%")
    
    if prediction == 1:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("🟢 Low Diabetes Risk")
    
    st.info("Model loaded from Pickle file")
