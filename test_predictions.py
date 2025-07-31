import pandas as pd
import numpy as np
import joblib
import os

print("Testing model predictions...")

# Load the balanced model
print("Loading balanced model...")
model = joblib.load("models/logreg_balanced.joblib")
scaler = joblib.load("models/scaler_balanced.joblib")
pca = joblib.load("models/pca_balanced.joblib")

# Load feature names
with open("models/feature_names_balanced.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Create a function to test with specific inputs
def test_prediction(input_data, description):
    print(f"\nTesting {description}...")
    
    # Create DataFrame with the correct feature order
    input_df = pd.DataFrame([input_data])
    
    # Ensure the DataFrame has all the required features in the correct order
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    # Standardize the input data
    input_scaled = scaler.transform(input_df)
    
    # Apply PCA transformation
    input_pca = pca.transform(input_scaled)
    
    # Make prediction
    prediction = model.predict(input_pca)
    
    # Get prediction probabilities
    proba = model.predict_proba(input_pca)
    
    result = "RAIN" if prediction[0] == 1 else "NO RAIN"
    
    print(f"Prediction: {result} (value: {prediction[0]})")
    print(f"Probability: No Rain={proba[0][0]:.4f}, Rain={proba[0][1]:.4f}")
    
    return prediction[0], proba[0]

# Test with high rainfall and high humidity (should predict rain)
rainy_input = {
    'MinTemp': 15.0,
    'MaxTemp': 25.0,
    'Rainfall': 20.0,
    'WindGustSpeed': 35.0,
    'WindSpeed9am': 20.0,
    'WindSpeed3pm': 25.0,
    'Humidity9am': 90.0,
    'Humidity3pm': 85.0,
    'Pressure9am': 1005.0,
    'Pressure3pm': 1000.0,
    'Temp9am': 18.0,
    'Temp3pm': 22.0,
    'RainToday_Yes': 1,
    'Month': 6,
    'Day': 15
}

# Test with low rainfall and low humidity (should predict no rain)
sunny_input = {
    'MinTemp': 20.0,
    'MaxTemp': 35.0,
    'Rainfall': 0.0,
    'WindGustSpeed': 25.0,
    'WindSpeed9am': 10.0,
    'WindSpeed3pm': 15.0,
    'Humidity9am': 40.0,
    'Humidity3pm': 30.0,
    'Pressure9am': 1020.0,
    'Pressure3pm': 1018.0,
    'Temp9am': 25.0,
    'Temp3pm': 32.0,
    'RainToday_Yes': 0,
    'Month': 1,
    'Day': 15
}

# Test predictions
rainy_pred, rainy_proba = test_prediction(rainy_input, "rainy conditions")
sunny_pred, sunny_proba = test_prediction(sunny_input, "sunny conditions")

print("\nSummary:")
print(f"Rainy conditions prediction: {'RAIN' if rainy_pred == 1 else 'NO RAIN'}")
print(f"Sunny conditions prediction: {'RAIN' if sunny_pred == 1 else 'NO RAIN'}")

if rainy_pred == 1 and sunny_pred == 0:
    print("\nSUCCESS: Model correctly predicts both rainy and sunny conditions!")
else:
    print("\nWARNING: Model may not be correctly distinguishing between rainy and sunny conditions.")
    
# Test with some additional scenarios
moderate_rain_input = {
    'MinTemp': 18.0,
    'MaxTemp': 28.0,
    'Rainfall': 5.0,
    'WindGustSpeed': 30.0,
    'WindSpeed9am': 15.0,
    'WindSpeed3pm': 20.0,
    'Humidity9am': 75.0,
    'Humidity3pm': 65.0,
    'Pressure9am': 1010.0,
    'Pressure3pm': 1008.0,
    'Temp9am': 22.0,
    'Temp3pm': 26.0,
    'RainToday_Yes': 1,
    'Month': 4,
    'Day': 10
}

borderline_input = {
    'MinTemp': 19.0,
    'MaxTemp': 29.0,
    'Rainfall': 2.0,
    'WindGustSpeed': 28.0,
    'WindSpeed9am': 12.0,
    'WindSpeed3pm': 18.0,
    'Humidity9am': 60.0,
    'Humidity3pm': 55.0,
    'Pressure9am': 1012.0,
    'Pressure3pm': 1010.0,
    'Temp9am': 23.0,
    'Temp3pm': 27.0,
    'RainToday_Yes': 0,
    'Month': 3,
    'Day': 20
}

test_prediction(moderate_rain_input, "moderate rain conditions")
test_prediction(borderline_input, "borderline conditions")