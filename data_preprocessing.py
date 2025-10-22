# data_preprocessing.py
# Author: Nethaji Machani
# CodSoft Internship ID: BY25RY228889
# Project: Credit Card Fraud Detection (Machine Learning)

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(train_path: str, test_path: str):
    """Load training and testing datasets."""
    print("🔹 Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Clean, select numeric features, and scale the data."""
    print("🔹 Checking for missing values...")
    print(f"Missing values in train data:\n{train_df.isnull().sum()}")
    print(f"Missing values in test data:\n{test_df.isnull().sum()}")

    target_col = "is_fraud"

    # Separate features and target
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    # Keep only numeric columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    print(f"🔹 Numeric columns used: {list(numeric_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols)

    print("✅ Data preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, save_dir="data/processed"):
    """Save processed data to disk."""
    os.makedirs(save_dir, exist_ok=True)
    X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)
    print(f"✅ Processed data saved to: {save_dir}")

if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    # Load, preprocess, and save data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_df, test_df)
    save_processed_data(X_train, X_test, y_train, y_test)
