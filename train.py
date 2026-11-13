import pandas as pd
import urllib.request
import zipfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Download and extract the UCI dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.request.urlretrieve(url, "sms.zip")
with zipfile.ZipFile("sms.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# 2. Load and clean the data
# The dataset is tab-separated with no headers
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Text vectorization (TF-IDF) and Stopword removal
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 4. Train the classification model (Naive Bayes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate and save the artifacts
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel (spam_model.pkl) and Vectorizer (vectorizer.pkl) exported successfully.")