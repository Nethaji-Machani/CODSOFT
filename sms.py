
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = pd.read_csv("spam.csv", encoding='latin-1')


df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("✅ Dataset loaded successfully!")
print(df.head())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['cleaned_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], df['label_num'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\n🔹 Training Models...")


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)


svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)

# 7️⃣ Model Evaluation
print("\n=========== 📊 Model Evaluation ===========")

print("\n📘 Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

print("\n📗 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n📙 Support Vector Machine Results:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


best_model = svm_model


joblib.dump(best_model, "spam_detector_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and TF-IDF Vectorizer saved successfully!")


example_sms = [
    "Congratulations! You won a $500 Amazon Gift Card. Click here!",
    "Hey, are we still going to the movie tonight?"
]
S
example_tfidf = vectorizer.transform(example_sms)
predictions = best_model.predict(example_tfidf)

for msg, pred in zip(example_sms, predictions):
    print(f"\nMessage: {msg}")
    print("Prediction:", "🚫 SPAM" if pred == 1 else "✅ LEGITIMATE")
