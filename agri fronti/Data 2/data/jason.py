import json

def get_estimated_npk(soil_type, last_crop, residue_left, rainfall_mm, data):
    for entry in data:
        if (
            entry["soil_type"] == soil_type and
            entry["last_crop"] == last_crop and
            entry["residue_left"] == residue_left and
            entry["rainfall_mm"] == rainfall_mm
        ):
            return {
                "N": entry["estimated_N"],
                "P": entry["estimated_P"],
                "K": entry["estimated_K"]
            }
    return None

# File path
n = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\crop_soil_weather_dataset.json"

# Load data
with open(n, "r") as file:
    dataset = json.load(file)

# Take user input
soil_type = input("Enter soil type (e.g., Loamy, Sandy, Clayey): ")
last_crop = input("Enter last crop (e.g., Wheat, Rice, Sugarcane): ")

# Convert string to boolean
residue_input = input("Was residue left on the field? (yes/no): ").strip().lower()
residue_left = True if residue_input in ["yes", "y"] else False

# Get rainfall input
try:
    rainfall_mm = float(input("Enter rainfall in mm (e.g., 225.05): "))
except ValueError:
    print("Invalid rainfall value.")
    exit()

# Find result
result = get_estimated_npk(soil_type, last_crop, residue_left, rainfall_mm, dataset)

# Display output
if result:
    print("→ Estimated NPK values:", result)
else:
    print("⚠️ No matching data found. Please check your inputs.")
