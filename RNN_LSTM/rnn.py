"""
=====================================================
Simple RNN Example
=====================================================

Author : Vasanth

A Recurrent Neural Network (RNN) processes
sequence data one step at a time.

Example:
I → Love → AI

=====================================================
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# -----------------------------
# Create Dummy Dataset
# -----------------------------

# Shape = (samples, time_steps, features)

X = np.random.random((100, 10, 1))
y = np.random.randint(0, 2, size=(100, 1))

print("Input Shape :", X.shape)
print("Output Shape:", y.shape)

# -----------------------------
# Build Model
# -----------------------------

model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        input_shape=(10, 1),
        activation="tanh"
    )
)

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
# Predict
# -----------------------------

prediction = model.predict(X[:5])

print("\nPredictions\n")

print(prediction)