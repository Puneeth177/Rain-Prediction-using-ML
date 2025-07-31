from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import joblib
import os

app = Flask(__name__, template_folder="template")

# Load the model and preprocessors
try:
    model = joblib.load("models/logreg_new.joblib")
    scaler = joblib.load("models/scaler_new.joblib")
    pca = joblib.load("models/pca_new.joblib")
    
    # Load feature names
    with open("models/feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print("Model and preprocessors loaded successfully")
except Exception as e:
    print(f"Error loading model and preprocessors: {e}")
    model = joblib.load("models/logreg.joblib")
    scaler = None
    pca = None
    feature_names = None
    print("Loaded original model as fallback")

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # DATE
            date = request.form['date']
            day = float(pd.to_datetime(date, format="%Y-%m-%d").day)
            month = float(pd.to_datetime(date, format="%Y-%m-%d").month)
            # MinTemp
            minTemp = float(request.form['mintemp'])
            # MaxTemp
            maxTemp = float(request.form['maxtemp'])
            # Rainfall
            rainfall = float(request.form['rainfall'])
            # Evaporation
            evaporation = float(request.form['evaporation'])
            # Sunshine
            sunshine = float(request.form['sunshine'])
            # Wind Gust Speed
            windGustSpeed = float(request.form['windgustspeed'])
            # Wind Speed 9am
            windSpeed9am = float(request.form['windspeed9am'])
            # Wind Speed 3pm
            windSpeed3pm = float(request.form['windspeed3pm'])
            # Humidity 9am
            humidity9am = float(request.form['humidity9am'])
            # Humidity 3pm
            humidity3pm = float(request.form['humidity3pm'])
            # Pressure 9am
            pressure9am = float(request.form['pressure9am'])
            # Pressure 3pm
            pressure3pm = float(request.form['pressure3pm'])
            # Temperature 9am
            temp9am = float(request.form['temp9am'])
            # Temperature 3pm
            temp3pm = float(request.form['temp3pm'])
            # Cloud 9am
            cloud9am = float(request.form['cloud9am'])
            # Cloud 3pm
            cloud3pm = float(request.form['cloud3pm'])
            # Location
            location = float(request.form['location'])
            # Wind Dir 9am
            windDir9am = float(request.form['winddir9am'])
            # Wind Dir 3pm
            windDir3pm = float(request.form['winddir3pm'])
            # Wind Gust Dir
            windGustDir = float(request.form['windgustdir'])
            # Rain Today
            rainToday = float(request.form['raintoday'])
            
            # Make prediction
            prediction = None
            
            # Try to use new model with scaler and PCA
            if scaler is not None and pca is not None and feature_names is not None:
                try:
                    # Create a DataFrame with the correct feature names
                    # This is a simplified approach - in a real application, you would need to handle
                    # categorical variables properly by creating the same one-hot encoded columns
                    
                    # Create a dictionary with all input values
                    input_data = {}
                    
                    # Map numeric features directly
                    numeric_features = {
                        'MinTemp': minTemp,
                        'MaxTemp': maxTemp,
                        'Rainfall': rainfall,
                        'WindGustSpeed': windGustSpeed,
                        'WindSpeed9am': windSpeed9am,
                        'WindSpeed3pm': windSpeed3pm,
                        'Humidity9am': humidity9am,
                        'Humidity3pm': humidity3pm,
                        'Pressure9am': pressure9am,
                        'Pressure3pm': pressure3pm,
                        'Temp9am': temp9am,
                        'Temp3pm': temp3pm,
                        'Month': month,
                        'Day': day
                    }
                    
                    # Initialize all features to 0
                    for feature in feature_names:
                        input_data[feature] = 0
                    
                    # Set the numeric features
                    for feature, value in numeric_features.items():
                        if feature in input_data:
                            input_data[feature] = value
                    
                    # Handle categorical features (simplified approach)
                    # RainToday
                    if rainToday == 1 and 'RainToday_Yes' in input_data:
                        input_data['RainToday_Yes'] = 1
                    
                    # Location (simplified - in reality, you'd need to map the location ID to the correct one-hot column)
                    location_id = int(location)
                    location_columns = [col for col in feature_names if col.startswith('Location_')]
                    if location_id < len(location_columns) and location_id >= 0:
                        input_data[location_columns[location_id]] = 1
                    
                    # Wind directions (simplified)
                    wind_dir_9am_columns = [col for col in feature_names if col.startswith('WindDir9am_')]
                    wind_dir_3pm_columns = [col for col in feature_names if col.startswith('WindDir3pm_')]
                    wind_gust_dir_columns = [col for col in feature_names if col.startswith('WindGustDir_')]
                    
                    wind_dir_9am_id = int(windDir9am)
                    wind_dir_3pm_id = int(windDir3pm)
                    wind_gust_dir_id = int(windGustDir)
                    
                    if wind_dir_9am_id < len(wind_dir_9am_columns) and wind_dir_9am_id >= 0:
                        input_data[wind_dir_9am_columns[wind_dir_9am_id]] = 1
                    
                    if wind_dir_3pm_id < len(wind_dir_3pm_columns) and wind_dir_3pm_id >= 0:
                        input_data[wind_dir_3pm_columns[wind_dir_3pm_id]] = 1
                    
                    if wind_gust_dir_id < len(wind_gust_dir_columns) and wind_gust_dir_id >= 0:
                        input_data[wind_gust_dir_columns[wind_gust_dir_id]] = 1
                    
                    # Create DataFrame with the correct feature order
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure the DataFrame has all the required features in the correct order
                    input_df = input_df[feature_names]
                    
                    # Standardize the input data
                    input_scaled = scaler.transform(input_df)
                    
                    # Apply PCA transformation
                    input_pca = pca.transform(input_scaled)
                    
                    # Make prediction
                    prediction = model.predict(input_pca)
                    print("Prediction made using new model with scaler and PCA")
                except Exception as e:
                    print(f"Error using new model: {e}")
            
            # Fallback to original approach if new model fails
            if prediction is None:
                # Use the original approach as fallback
                input_lst = [location, rainfall, sunshine, windGustSpeed, humidity3pm, 
                            pressure3pm, cloud3pm, temp3pm, rainToday, month]
                input_array = np.array(input_lst).reshape(1, -1)
                prediction = model.predict(input_array)
                print("Prediction made using original model")
            
            # Print prediction for debugging
            print(f"Prediction result: {prediction}")
            
            # Determine output
            output = prediction[0]  # Get the first element of the prediction array
            
            if output == 0:
                return render_template("after_sunny.html")
            else:
                return render_template("after_rainy.html")
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return render_template("predictor.html", error=str(e))
            
    return render_template("predictor.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)