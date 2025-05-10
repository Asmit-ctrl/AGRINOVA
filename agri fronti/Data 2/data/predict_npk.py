import json
import numpy as np
import tensorflow as tf
import joblib
from fuzzywuzzy import process

dataset_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\crop_soil_weather_dataset.json"
with open(dataset_path, "r") as file:
    dataset = json.load(file)

def suggest_closest_match(user_input, valid_options):
    closest_match = process.extractOne(user_input, valid_options)
    return closest_match[0] if closest_match else None

def is_valid_combination(soil_type, last_crop):
    combinations = set()
    for entry in dataset:
        combinations.add((entry['soil_type'], entry['last_crop']))
    return (soil_type, last_crop) in combinations, combinations

def predict_npk(soil_type, last_crop, residue_left, rainfall_mm, temperature, humidity):
    encoder = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\encoder.pkl")
    scaler = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\scaler.pkl")
    model = tf.keras.models.load_model("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\npk_model.keras")

    is_valid, combinations = is_valid_combination(soil_type, last_crop)
    if not is_valid:
        return "invalid_combination", combinations

    cat_input = encoder.transform([[soil_type, last_crop]])
    residue_input = int(residue_left)
    num_input = np.array([[residue_input, rainfall_mm, temperature, humidity]])
    combined_input = np.concatenate([cat_input, num_input], axis=1)
    scaled_input = scaler.transform(combined_input)

    prediction = model.predict(scaled_input)

    return {
        "estimated_N": round(prediction[0][0], 2),
        "estimated_P": round(prediction[0][1], 2),
        "estimated_K": round(prediction[0][2], 2)
    }

if __name__ == "__main__":
    soil_type_input = input("Enter soil type (e.g., Sandy, Loamy, Clayey): ").strip().capitalize()
    last_crop_input = input("Enter last crop type (e.g., Wheat, Rice, Maize): ").strip().capitalize()

    residue_input = input("Enter residue left (True/False): ").strip().lower()
    if residue_input not in ['true', 'false']:
        print("❌ Error: Please enter 'True' or 'False' for residue left.")
        exit()
    residue_left = True if residue_input == 'true' else False

    try:
        rainfall_mm = float(input("Enter rainfall in mm: ").strip())
        temperature = float(input("Enter temperature in °C: ").strip())
        humidity = float(input("Enter humidity in %: ").strip())
    except ValueError:
        print("❌ Error: Please enter valid numeric values for rainfall, temperature, and humidity.")
        exit()

    soil_type_input = suggest_closest_match(soil_type_input, [entry['soil_type'] for entry in dataset]) or soil_type_input
    last_crop_input = suggest_closest_match(last_crop_input, [entry['last_crop'] for entry in dataset]) or last_crop_input

    result = predict_npk(soil_type_input, last_crop_input, residue_left, rainfall_mm, temperature, humidity)

    if isinstance(result, tuple) and result[0] == "invalid_combination":
        print(f"\n❌ Error: The combination of soil type '{soil_type_input}' and last crop '{last_crop_input}' is not valid in the dataset.")
        print("\n✅ Available combinations:")
        for s, c in sorted(result[1]):
            print(f"  - {s} - {c}")
        print()
    else:
        print("\n------------------- Prediction Results -------------------")
        print(f"Estimated N (Nitrogen): \t{result['estimated_N']} kg/ha")
        print(f"Estimated P (Phosphorus): \t{result['estimated_P']} kg/ha")
        print(f"Estimated K (Potassium): \t{result['estimated_K']} kg/ha")
        print("-----------------------------------------------------------\n")