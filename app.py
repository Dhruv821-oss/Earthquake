import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("preprocessor.pkl")

st.title("Earthquake Magnitude Predictor 🌍")

st.write("Enter the parameters below:")

# Minimal user input
latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)
depth = st.number_input("Depth (km)", value=10.0)

input_data = np.array([[latitude, longitude, depth]])

if st.button("Predict Magnitude"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.success(f"Predicted Earthquake Magnitude: {prediction[0]:.2f}")
