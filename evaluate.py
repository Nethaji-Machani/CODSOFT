# evaluate.py
# Author: Nethaji Machani
# CodSoft Internship ID: BY25RY228889
# Project: Credit Card Fraud Detection (Machine Learning)

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib
import matplotlib.pyplot as plt
import os


def evaluate_model(model_path="models/best_model.pkl", data_dir="data/processed", save_results=True):
    # Load processed test data
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    # Load trained model
    model = joblib.load(model_path)
    print(f"🔹 Loaded model: {model}")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    if save_results:
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig("results/figures/confusion_matrix.png")
        print("✅ Confusion matrix saved to results/figures/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()
