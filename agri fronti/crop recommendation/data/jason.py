import json

input_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\datasets_mp\\mp_crop_recommendation_numeric_npk.json"
output_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\datasets_mp\\mp2_crop_recommendation_numeric_npk.json"

with open(input_path, 'r') as file:
    data = json.load(file)

# Flatten the npk_estimate field into top-level N, P, K
for entry in data:
    if "npk_estimate" in entry:
        npk = entry.pop("npk_estimate")
        entry["N"] = npk.get("N", None)
        entry["P"] = npk.get("P", None)
        entry["K"] = npk.get("K", None)

# Save the cleaned data
with open(output_path, 'w') as file:
    json.dump(data, file, indent=2)

print("✅ NPK values have been flattened and saved to:", output_path)
