import numpy as np
import tensorflow as tf
import joblib

def predict_npk(soil_type, last_crop, residue_left, rainfall_mm):
    encoder = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\encoder.pkl")
    scaler = joblib.load("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\scaler.pkl")
    model = tf.keras.models.load_model("D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\npk_model.keras")

    cat_input = encoder.transform([[soil_type, last_crop]])
    residue_input = int(residue_left)
    num_input = np.array([[residue_input, rainfall_mm]])
    combined_input = np.concatenate([cat_input, num_input], axis=1)
    scaled_input = scaler.transform(combined_input)

    prediction = model.predict(scaled_input)

    return {
        "estimated_N": round(prediction[0][0], 2),
        "estimated_P": round(prediction[0][1], 2),
        "estimated_K": round(prediction[0][2], 2)
    }

if __name__ == "__main__":
    soil_type = input("Enter soil type (e.g., sandy, loamy, clay): ").strip().lower()
    last_crop = input("Enter last crop type (e.g., wheat, rice, maize): ").strip().lower()
    residue_left = input("Enter residue left (True/False): ").strip().lower()
    residue_left = True if residue_left == 'true' else False
    rainfall_mm = float(input("Enter rainfall in mm: ").strip())

    result = predict_npk(soil_type, last_crop, residue_left, rainfall_mm)

    print("\n------------------- Prediction Results -------------------")
    print(f"Estimated N (Nitrogen): \t\t{result['estimated_N']} kg/ha")
    print(f"Estimated P (Phosphorus): \t{result['estimated_P']} kg/ha")
    print(f"Estimated K (Potassium): \t{result['estimated_K']} kg/ha")
    print("-----------------------------------------------------------\n")
