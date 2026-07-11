import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("dataset/IMDB_Dataset.csv")

# Convert labels
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Features
X = df["review"]

y = df["sentiment"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Training Completed")