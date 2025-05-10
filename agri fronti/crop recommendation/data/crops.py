import json
import pandas as pd
import numpy as np
import joblib
import os
import re
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, MultiLabelBinarizer

# Load trained model
model_path = "crop recommendation\\data\\trainedmodels\\crop_recommendation_sequential_model.keras"
assert os.path.exists(model_path), "❌ Model file not found!"
model = load_model(model_path)
print("✅ Model loaded successfully.")

# Load preprocessors
encoder_path = "crop recommendation\\data\\trainedmodels\\encoder_recommended.pkl"
scaler_path = "crop recommendation\\data\\trainedmodels\\scaler_recommended.pkl"
crop_mlb_path = "crop recommendation\\data\\trainedmodels\\crop_mlb_recommended.pkl"
soil_encoder_path = "crop recommendation\\data\\trainedmodels\\soil_encoder_recommended.pkl"

encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)
crop_mlb = joblib.load(crop_mlb_path)
soil_encoder = joblib.load(soil_encoder_path)
print("✅ Preprocessors loaded successfully.")

# Example input data
new_data = {
    "region": "Sehore, MP",
    "weather": "Semi-Arid",
    "season": "Kharif",
    "rainfall": 500,
    "N": 82.35,
    "P": 44.85,
    "K": 112.8
}

# Convert dictionary to DataFrame
new_df = pd.DataFrame([new_data])

# One-hot encode categorical features
X_cat_new = encoder.transform(new_df[['region', 'weather', 'season']])
X_cat_new = pd.DataFrame(X_cat_new, columns=encoder.get_feature_names_out(['region', 'weather', 'season']))

# Combine categorical and numeric features
X_new = pd.concat([X_cat_new, new_df[['N', 'P', 'K', 'rainfall']]], axis=1)

# Scale features
X_new_scaled = scaler.transform(X_new)
print("✅ New input data processed successfully.")

# Make predictions
soil_pred, crops_pred = model.predict(X_new_scaled)

# Ensure correct shape
soil_pred = np.array(soil_pred).reshape(1, -1)
crops_pred = np.array(crops_pred).reshape(1, -1)
print(f"🔍 Soil Prediction Shape: {soil_pred.shape}")
print(f"🔍 Crop Prediction Shape: {crops_pred.shape}")

# Decode soil prediction
predicted_soil = soil_encoder.inverse_transform([np.argmax(soil_pred)])

# Adjust threshold for crops prediction
threshold = 0.64  # Adjusted to be more precise in recommendations
crop_probabilities = crops_pred[0]  # Extract probabilities for each crop

# Filter out crops that have probabilities slightly above the threshold
filtered_crops = [
    crop for crop, prob in zip(crop_mlb.classes_, crop_probabilities)
    if prob > threshold and prob < 0.95  # Exclude overly confident but incorrect crops
]

# Display raw probability scores for debugging
print("🔍 Crop Prediction Probabilities:")
for crop, prob in zip(crop_mlb.classes_, crop_probabilities):
    print(f"{crop}: {prob:.4f}")

# Final predictions
print(f"🌱 Predicted Soil Type: {predicted_soil[0]}")
print(f"🌾 Recommended Crops: {filtered_crops}")