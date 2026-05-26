import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

df = pd.read_csv("weather.csv")
df.columns = df.columns.str.strip()

X = df[['temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]

# 🔥 FIX HERE (IMPORTANT)
y = df['Rain'].str.lower().str.strip()
y = y.map({'rain': 1, 'no rain': 0})

# remove missing values (safe step)
df = df.dropna()
X = df[['temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = df['Rain'].str.lower().str.strip().map({'rain': 1, 'no rain': 0})

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/weather_model.pkl")

print("✅ Model trained successfully!")
print("Accuracy:", model.score(X_test, y_test))