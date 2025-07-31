import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

print("Starting preprocessing objects creation...")

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

try:
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('weatherAUS.csv')
    
    # Drop columns with too many missing values
    df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1, inplace=True)
    
    # Drop the Date column as it's not needed for the model
    df.drop(['Date'], axis=1, inplace=True)
    
    # Handle missing values using mean for numerical columns and most frequent for categorical columns
    print("Handling missing values...")
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Apply imputers
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Ensure the target column exists and is correctly encoded if necessary
    if 'RainTomorrow' in df.columns:
        df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode all categorical columns
    print("One-hot encoding categorical columns...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    df.to_csv('preprocessed_1.csv', index=False)
    
    # Split the dataset
    X = df.drop(['RainTomorrow'], axis=1)
    y = df['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the scaler
    print("Creating and fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and fit the PCA
    print("Creating and fitting PCA...")
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Save the scaler and PCA
    print("Saving scaler and PCA...")
    joblib.dump(scaler, './models/scaler.joblib')
    joblib.dump(pca, './models/pca.joblib')
    
    print("Successfully created and saved preprocessing objects!")
    
except Exception as e:
    print(f"Error creating preprocessing objects: {e}")