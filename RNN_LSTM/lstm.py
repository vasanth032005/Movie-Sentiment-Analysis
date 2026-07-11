"""
=====================================================
LSTM Example
=====================================================

Author : Vasanth

LSTM remembers information for long sequences.

Used in:
✔ Sentiment Analysis
✔ Translation
✔ Chatbots
✔ Speech Recognition

=====================================================
"""

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Dummy Dataset
# -----------------------------

X = np.random.random((100, 20, 1))
y = np.random.randint(0, 2, (100, 1))

print("Input Shape :", X.shape)

# -----------------------------
# Build Model
# -----------------------------

model = Sequential()

model.add(
    LSTM(
        units=64,
        input_shape=(20, 1)
    )
)

model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary\n")

model.summary()

# -----------------------------
# Train
# -----------------------------

model.fit(
    X,
    y,
    epochs=5,
    batch_size=8
)

# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(X[:3])

print("\nPrediction")

print(prediction)