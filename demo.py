import joblib
import pandas as pd

# Model load
model = joblib.load("models/weather_model.pkl")

# User input
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))

# DataFrame create
data = pd.DataFrame({
    "temperature": [temp],
    "humidity": [humidity]
})

# Prediction
prediction = model.predict(data)

# Result
print("Predicted Weather:", prediction[0])