import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

print("Starting balanced model creation...")

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

try:
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('weatherAUS.csv')
    
    # Drop columns with too many missing values
    df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1, inplace=True)
    
    # Extract day and month from Date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
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
    
    # Balance the dataset
    print("Balancing the dataset...")
    df_majority = df[df['RainTomorrow'] == 0]
    df_minority = df[df['RainTomorrow'] == 1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=42)  # reproducible results
    
    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # One-hot encode all categorical columns
    print("One-hot encoding categorical columns...")
    categorical_cols = df_balanced.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        df_balanced = pd.get_dummies(df_balanced, columns=categorical_cols, drop_first=True)
    
    # Save preprocessed data
    print("Saving preprocessed balanced data...")
    df_balanced.to_csv('preprocessed_balanced.csv', index=False)
    
    # Split the dataset
    X = df_balanced.drop(['RainTomorrow'], axis=1)
    y = df_balanced['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the scaler
    print("Creating and fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and fit the PCA
    print("Creating and fitting PCA...")
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train the logistic regression model with balanced class weights
    print("Training logistic regression model with balanced weights...")
    log_reg = LogisticRegression(random_state=42, class_weight='balanced')
    log_reg.fit(X_train_pca, y_train)
    
    # Save the model, scaler, and PCA
    print("Saving model, scaler, and PCA...")
    joblib.dump(log_reg, './models/logreg_balanced.joblib')
    joblib.dump(scaler, './models/scaler_balanced.joblib')
    joblib.dump(pca, './models/pca_balanced.joblib')
    
    # Save the column names for reference
    with open('./models/feature_names_balanced.txt', 'w') as f:
        f.write('\n'.join(X.columns))
    
    print("Successfully created and saved balanced model and preprocessing objects!")
    
except Exception as e:
    print(f"Error creating balanced model: {e}")