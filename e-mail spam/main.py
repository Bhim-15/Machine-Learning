
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

nltk.download('stopwords')


df = pd.read_csv(r"C:\Users\DELL-PC\Desktop\ML practical\house price prediction\housing.csv")



df = df[['text', 'label_num']]
print("✅ Dataset Loaded Successfully!")
print(df.head())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = " ".join(word for word in text.split() if word not in stopwords.words('english'))
    return text

df['text'] = df['text'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_num'], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=3000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vect, y_train)


y_pred = model.predict(X_test_vect)
print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


sample = ["Hi John, can we reschedule our meeting to next Tuesday"]
sample_vect = vectorizer.transform(sample)
prediction = model.predict(sample_vect)[0]
print("\n🧾 Test Message:", sample[0])
print("Prediction:", "Spam 🚫" if prediction == 1 else "Not Spam ✅")
