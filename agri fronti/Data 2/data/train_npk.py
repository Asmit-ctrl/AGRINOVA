import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import joblib

# File path
n = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\crop_soil_weather_dataset.json"

# Load data
with open(n, "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[["soil_type", "last_crop"]])
cat_feature_names = encoder.get_feature_names_out(["soil_type", "last_crop"])
cat_df = pd.DataFrame(cat_features, columns=cat_feature_names)

df["residue_left"] = df["residue_left"].astype(int)
numerical = df[["residue_left", "rainfall_mm"]].reset_index(drop=True)

X = pd.concat([cat_df, numerical], axis=1)
y = df[["estimated_N", "estimated_P", "estimated_K"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3)  # N, P, K outputs
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

model.save("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\npk_model.keras")
joblib.dump(encoder, "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\encoder.pkl")
joblib.dump(scaler, "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\scaler.pkl")

print("✅ Model and preprocessors saved!")