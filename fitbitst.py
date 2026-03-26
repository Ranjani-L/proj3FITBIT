import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("pipeline.pkl", "rb"))

st.title("FITBIT Calorie Burn Predictor")
st.image("C:\\Users\\Admin\\Desktop\\env\\Scripts\\calories.jpg")
st.write("Enter your workout details to estimate calories burned")

# Inputs
age = st.number_input("Age")
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)")
height = st.number_input("Height (m)")
max_bpm = st.number_input("Max BPM")
avg_bpm = st.number_input("Avg BPM")
resting_bpm = st.number_input("Resting BPM")
Session_Duration = st.number_input("Session_Duration (hours)")
workout_type = st.selectbox("Workout_Type", ["Cardio", "Strength", "HIIT", "Yoga"])
fat = st.number_input("Fat_Percentage")
water = st.number_input("Water_Intake")
frequency = st.number_input("Workout_Frequency")
experience = st.selectbox("Experience_Level", ["0","1","2","3"])
bmi = st.number_input("BMI")
baseMET = st.number_input("Base MET")
HR_Intensity = st.number_input("Heart Rate Intensity")
Effective_MET = st.number_input("Effective MET")

# Encoding
gender = 0 if gender == "Male" else 1
workout_type = {"Cardio": 0, "Strength": 1, "HIIT": 2, "Yoga": 3}[workout_type]

# Predict
if st.button("Predict"):
    input_data = [age, gender, weight, height, max_bpm, avg_bpm,
                  resting_bpm, Session_Duration, workout_type, fat, water,
                  frequency, experience,bmi,baseMET,HR_Intensity, Effective_MET]
    input_data = np.asarray(input_data).reshape(1, -1)

    prediction = model.predict(input_data)

    st.success(f"🔥 Calories Burned: {prediction[0]:.2f}")