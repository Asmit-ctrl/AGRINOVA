import json
import pandas as pd
import numpy as np
import joblib
import random
import os
import re
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from scipy.spatial import KDTree

# Load dataset
dataset_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\datasets_mp\\mp_crop_recommendation_numeric_npk.json"
with open(dataset_path, 'r') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)
assert not df.empty, "❌ DataFrame is empty, check dataset path or format!"
print(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# Extract numeric rainfall from string range
def extract_rainfall_value(rainfall_str):
    match = re.match(r'(\d+)-(\d+)', rainfall_str)
    if match:
        return random.randint(int(match.group(1)), int(match.group(2)))
    return np.nan

df['rainfall'] = df['rainfall'].apply(lambda x: extract_rainfall_value(x.split('(')[1].split(')')[0]) if '(' in x else np.nan)
assert df['rainfall'].notna().sum() > 0, "❌ Rainfall extraction failed!"
print("✅ Rainfall extraction complete.")

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(df[['region', 'weather', 'season']])
X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(['region', 'weather', 'season']))
assert X_cat.shape[1] > 0, "❌ Encoding failed!"
print("✅ Encoding successful.")

# Create an NPK lookup tree for closest match retrieval
npk_data = df[['N', 'P', 'K']].values
npk_tree = KDTree(npk_data)

def get_closest_npk(n, p, k):
    _, closest_index = npk_tree.query([n, p, k])
    return df.iloc[closest_index][['N', 'P', 'K']].values

# Encode targets
soil_encoder = LabelEncoder()
y_soil = soil_encoder.fit_transform(df['soil_type'])

crop_mlb = MultiLabelBinarizer()
y_crops = crop_mlb.fit_transform(df['recommended_crops'])

# Combine categorical and numeric features
X = pd.concat([X_cat, df[['N', 'P', 'K', 'rainfall']]], axis=1)

# Split data
X_train, X_test, y_train_soil, y_test_soil, y_train_crops, y_test_crops = train_test_split(
    X, y_soil, y_crops, test_size=0.2, random_state=42)

assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "❌ Data splitting failed!"
print("✅ Data split verified.")

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build optimized model using Functional API
input_layer = Input(shape=(X_train_scaled.shape[1],))
hidden_layer = Dense(128, activation='relu')(input_layer)
hidden_layer = BatchNormalization()(hidden_layer)
hidden_layer = Dropout(0.3)(hidden_layer)
hidden_layer = Dense(64, activation='relu')(hidden_layer)
hidden_layer = BatchNormalization()(hidden_layer)
hidden_layer = Dropout(0.3)(hidden_layer)
hidden_layer = Dense(32, activation='relu')(hidden_layer)

soil_type_output = Dense(len(np.unique(y_soil)), activation='softmax', name='soil_type_output')(hidden_layer)
recommended_crops_output = Dense(y_crops.shape[1], activation='sigmoid', name='recommended_crops_output')(hidden_layer)

model = Model(inputs=input_layer, outputs=[soil_type_output, recommended_crops_output])
model.summary()
print("✅ Model compiled successfully.")

# Compile model with improved loss weighting
model.compile(
    optimizer='adam',
    loss={
        'soil_type_output': 'sparse_categorical_crossentropy',
        'recommended_crops_output': 'binary_crossentropy'
    },
    loss_weights={'soil_type_output': 1.2, 'recommended_crops_output': 0.8},  # Adjusted for balanced learning
    metrics={
        'soil_type_output': 'accuracy',
        'recommended_crops_output': 'accuracy'
    }
)

# Train model with improved validation
print("Training the model...")
history = model.fit(
    X_train_scaled,
    {
        'soil_type_output': y_train_soil,
        'recommended_crops_output': y_train_crops
    },
    epochs=150,  # Increased epochs for better convergence
    batch_size=16,  # Adjusted batch size for stable training
    validation_split=0.2,
    verbose=1
)

# Save model
model_save_path = "crop_recommendation_optimized_model.keras"
model.save(model_save_path)
assert os.path.exists(model_save_path), "❌ Model file not saved!"
print(f"✅ Model saved successfully as '{model_save_path}'.")

# Save preprocessors
encoder_save_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\encoder_recommended.pkl"
scaler_save_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\scaler_recommended.pkl"
crop_mlb_save_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\crop_mlb_recommended.pkl"
soil_encoder_save_path = "D:\\projectllp\\logathon\\agri fronti\\crop_recommendation\\data\\trainedmodels\\soil_encoder_recommended.pkl"

for path, obj in zip(
    [encoder_save_path, scaler_save_path, crop_mlb_save_path, soil_encoder_save_path],
    [encoder, scaler, crop_mlb, soil_encoder]
):
    joblib.dump(obj, path)
    assert os.path.exists(path), f"❌ Missing file: {path}"
    print(f"✅ Preprocessor saved: {path}")

print("🎯 Model training complete and all files saved successfully.")