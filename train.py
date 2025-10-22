# train.py
# Author: Nethaji Machani
# CodSoft Internship ID: BY25RY228889
# Project: Credit Card Fraud Detection (Machine Learning)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# SMOTE for oversampling
from imblearn.over_sampling import SMOTE

def train_models():
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Apply SMOTE to balance training data
    print("🔹 Applying SMOTE to balance classes in training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE, training shape: {X_train_res.shape}, {y_train_res.shape}")

    # Create models with balanced class weight
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }

    best_model_name = None
    best_model_score = 0
    best_model = None

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=4)
        print(f"\nClassification Report for {name}:\n{report}\n")

        # Use F1-score for fraud class (class 1) as metric to select best model
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, pos_label=1)
        if f1 > best_model_score:
            best_model_score = f1
            best_model = model
            best_model_name = name

    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/best_model.pkl")
    print(f"✅ Best model saved: {best_model_name} with F1-score for fraud class: {best_model_score:.4f}")

if __name__ == "__main__":
    train_models()
