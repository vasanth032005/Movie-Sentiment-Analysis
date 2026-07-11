"""
=========================================================
Model Evaluation for Sentiment Analysis
=========================================================

Author : Vasanth

This script:
1. Loads the IMDB dataset
2. Loads the trained model
3. Makes predictions
4. Calculates evaluation metrics

=========================================================
"""

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# Parameters
# ---------------------------------------

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# ---------------------------------------
# Load Dataset
# ---------------------------------------

(_, _), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post"
)

# ---------------------------------------
# Load Model
# ---------------------------------------

model = load_model("sentiment_model.keras")

# ---------------------------------------
# Prediction
# ---------------------------------------

prediction = model.predict(X_test)

prediction = (prediction > 0.5).astype(int)

# ---------------------------------------
# Metrics
# ---------------------------------------

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(y_test, prediction)

recall = recall_score(y_test, prediction)

f1 = f1_score(y_test, prediction)

print("="*60)
print("Model Evaluation")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, prediction))

print("\nClassification Report\n")

print(classification_report(y_test, prediction))