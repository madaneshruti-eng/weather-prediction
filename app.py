import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("/models/weather_model.pkl")

st.title("🌦️ Weather Prediction (ML Model)")
st.write("Fill weather parameters to predict Rain or No Rain")

# Inputs
temperature = st.number_input("Temperature", value=25)
humidity = st.number_input("Humidity", value=60)
wind_speed = st.number_input("Wind Speed", value=10)
cloud_cover = st.number_input("Cloud Cover", value=50)
pressure = st.number_input("Pressure", value=1010)

# Predict button
if st.button("Predict Weather"):
    input_data = pd.DataFrame([[
        temperature,
        humidity,
        wind_speed,
        cloud_cover,
        pressure
    ]], columns=[
        "temperature",
        "Humidity",
        "Wind_Speed",
        "Cloud_Cover",
        "Pressure"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🌧️ Rain Expected")
    else:
        st.success("☀️ No Rain")
