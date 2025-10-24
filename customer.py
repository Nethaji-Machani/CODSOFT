import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from joblib import dump
data = pd.read_csv("churn_modelling.csv")

print("Data shape:", data.shape)
print(data.head())

data = data.drop(columns=["RowNumber", "CustomerId", "Surname"])

le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Geography"] = le.fit_transform(data["Geography"])


X = data.drop("Exited", axis=1)
y = data["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probas)
    }


results_df = pd.DataFrame(results).T.sort_values(by="ROC-AUC", ascending=False)
print("\n=== Model Performance ===")
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name}")


y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

dump(best_model, "best_churn_model.joblib")
print("💾 Model saved as 'best_churn_model.joblib'")
