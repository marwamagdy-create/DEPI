import streamlit as st
import pandas as pd
import mlflow.pyfunc


st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes risk using your trained MLflow model.")

# -----------------------------
# Load MLflow Model
# -----------------------------
RUN_ID = "b5f877d9c5ca4c56a069a75a81304c8b"
MODEL_URI = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

# -----------------------------
# User Input
# -----------------------------
st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hbA1c = st.number_input("HbA1c Level", min_value=3.5, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

# - one-hot encoding
gender = st.selectbox("Gender", ["Male", "Female"])
gender_Female = 1 if gender == "Female" else 0
gender_Male = 1 if gender == "Male" else 0

#  - one-hot encoding
race = st.selectbox("Race", ["AfricanAmerican","Asian","Caucasian","Hispanic","Other"])
race_dict = {r: int(r==race) for r in ["AfricanAmerican","Asian","Caucasian","Hispanic","Other"]}

#  - one-hot encoding
smoking = st.selectbox("Smoking History", ["Current", "Ever", "Former", "Never", "Not current", "No Info"])
smoking_dict = {f"smoking_history_{s.lower().replace(' ', '_')}": int(s==smoking) for s in ["Current","Ever","Former","Never","Not current","No Info"]}


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

columns_order = model.metadata.get_input_schema().input_names
input_df = pd.DataFrame([input_dict], columns=columns_order)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("📊 Prediction Result")
    st.write(f"**Diabetes Prediction:** {prediction}")
    if prediction == 1:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("🟢 Low Diabetes Risk")
    st.info(f"Model loaded from MLflow run: {RUN_ID}")

