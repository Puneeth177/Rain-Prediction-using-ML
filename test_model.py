import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the balanced model
print("Loading balanced model...")
model = joblib.load("models/logreg_balanced.joblib")
scaler = joblib.load("models/scaler_balanced.joblib")
pca = joblib.load("models/pca_balanced.joblib")

# Load feature names
with open("models/feature_names_balanced.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Load the preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('preprocessed_balanced.csv')

# Split into features and target
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# Ensure X has the same columns as feature_names
X = X[feature_names]

# Apply preprocessing
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_pca)

# Evaluate the model
print("Evaluating model...")
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Check class distribution in predictions
print("\nClass distribution in predictions:")
print(pd.Series(y_pred).value_counts())

# Check class distribution in actual data
print("\nClass distribution in actual data:")
print(pd.Series(y).value_counts())

# Test with some specific inputs
print("\nTesting with specific inputs...")

# Create a function to test with specific inputs
def test_prediction(input_data):
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
rainy_pred, rainy_proba = test_prediction(rainy_input)
sunny_pred, sunny_proba = test_prediction(sunny_input)

print(f"Rainy conditions prediction: {rainy_pred} (0=No Rain, 1=Rain)")
print(f"Rainy conditions probability: No Rain={rainy_proba[0]:.4f}, Rain={rainy_proba[1]:.4f}")

print(f"Sunny conditions prediction: {sunny_pred} (0=No Rain, 1=Rain)")
print(f"Sunny conditions probability: No Rain={sunny_proba[0]:.4f}, Rain={sunny_proba[1]:.4f}")

# Check model coefficients
print("\nModel coefficients:")
print(model.coef_)
print("\nModel intercept:")
print(model.intercept_)

# Force a prediction of 1 (Rain)
print("\nForcing a prediction of Rain (1)...")
# Modify the model to always predict rain
model.intercept_ = np.array([10.0])  # Large positive intercept to bias towards class 1
joblib.dump(model, "models/logreg_always_rain.joblib")
print("Modified model saved as 'logreg_always_rain.joblib'")