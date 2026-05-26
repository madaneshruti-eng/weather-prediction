from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/weather_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        temp = float(request.form.get("temperature"))
        humidity = float(request.form.get("Humidity"))
        wind = float(request.form.get("Wind_Speed"))
        cloud = float(request.form.get("Cloud_Cover"))
        pressure = float(request.form.get("Pressure"))

        features = np.array([[temp, humidity, wind, cloud, pressure]])

        prediction = model.predict(features)

        result = "Rain" if prediction[0] == 1 else "No Rain"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)