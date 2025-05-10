import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import joblib
from fuzzywuzzy import process

n = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\datasets_mp\\crop_soil_weather_dataset.json"

with open(n, "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[["soil_type", "last_crop"]])
cat_feature_names = encoder.get_feature_names_out(["soil_type", "last_crop"])
cat_df = pd.DataFrame(cat_features, columns=cat_feature_names)

df["residue_left"] = df["residue_left"].astype(int)
numerical = df[["residue_left", "rainfall_mm", "temperature_C", "humidity_percent"]].reset_index(drop=True)

X = pd.concat([cat_df, numerical], axis=1)
y = df[["estimated_N", "estimated_P", "estimated_K"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

model.save("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\npk_model.keras")
joblib.dump(encoder, "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\encoder.pkl")
joblib.dump(scaler, "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\scaler.pkl")

print("✅ Model and preprocessors saved!")

def suggest_closest_match(user_input, valid_options):
    closest_match = process.extractOne(user_input, valid_options)
    return closest_match[0] if closest_match else None

def predict_npk(soil_type, last_crop, residue_left, rainfall_mm, temperature_C, humidity_percent):
    encoder = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\encoder.pkl")
    scaler = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\scaler.pkl")
    model = tf.keras.models.load_model("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\npk_model.keras")

    valid_soil_types = df['soil_type'].unique().tolist()
    valid_crop_types = df['last_crop'].unique().tolist()

    soil_type = suggest_closest_match(soil_type, valid_soil_types) or soil_type
    last_crop = suggest_closest_match(last_crop, valid_crop_types) or last_crop

    if soil_type not in valid_soil_types or last_crop not in valid_crop_types:
        return f"❌ Error: The combination of soil type '{soil_type}' and last crop '{last_crop}' is not valid."

    cat_input = encoder.transform([[soil_type, last_crop]])
    residue_input = int(residue_left)
    num_input = np.array([[residue_input, rainfall_mm, temperature_C, humidity_percent]])
    combined_input = np.concatenate([cat_input, num_input], axis=1)
    scaled_input = scaler.transform(combined_input)

    prediction = model.predict(scaled_input)

    return {
        "estimated_N": round(prediction[0][0], 2),
        "estimated_P": round(prediction[0][1], 2),
        "estimated_K": round(prediction[0][2], 2)
    }

if __name__ == "__main__":
    soil_type_input = input("Enter soil type (e.g., sandy, loamy, clay): ").strip().lower()
    last_crop_input = input("Enter last crop type (e.g., wheat, rice, maize): ").strip().lower()

    residue_input = input("Enter residue left (True/False): ").strip().lower()
    if residue_input not in ['true', 'false']:
        print("Error: Please enter 'True' or 'False' for residue left.")
        exit()
    residue_left = True if residue_input == 'true' else False

    try:
        rainfall_mm = float(input("Enter rainfall in mm: ").strip())
        temperature_C = float(input("Enter temperature in Celsius: ").strip())
        humidity_percent = float(input("Enter humidity percentage: ").strip())
    except ValueError:
        print("Error: Please enter valid numbers for rainfall, temperature, and humidity.")
        exit()

    result = predict_npk(soil_type_input, last_crop_input, residue_left, rainfall_mm, temperature_C, humidity_percent)

    if isinstance(result, str):
        print(result)
    else:
        print("\n------------------- Prediction Results -------------------")
        print(f"Estimated N (Nitrogen): \t{result['estimated_N']} kg/ha")
        print(f"Estimated P (Phosphorus): \t{result['estimated_P']} kg/ha")
        print(f"Estimated K (Potassium): \t{result['estimated_K']} kg/ha")
        print("-----------------------------------------------------------\n")
