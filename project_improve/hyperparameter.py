"""
=========================================================
Hyperparameter Tuning Demo
=========================================================

Author : Vasanth

This script compares different
LSTM configurations.

=========================================================
"""

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# ------------------------------------
# Function
# ------------------------------------

def build_model(
        embedding_dim,
        lstm_units,
        dropout_rate):

    model = Sequential()

    model.add(
        Embedding(
            VOCAB_SIZE,
            embedding_dim,
            input_length=MAX_LENGTH
        )
    )

    model.add(
        LSTM(lstm_units)
    )

    model.add(
        Dropout(dropout_rate)
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------
# Model 1
# ------------------------------------

model1 = build_model(
    embedding_dim=64,
    lstm_units=32,
    dropout_rate=0.2
)

print("="*60)
print("Model 1")
print("="*60)

model1.summary()

# ------------------------------------
# Model 2
# ------------------------------------

model2 = build_model(
    embedding_dim=128,
    lstm_units=64,
    dropout_rate=0.3
)

print("="*60)
print("Model 2")
print("="*60)

model2.summary()

# ------------------------------------
# Model 3
# ------------------------------------

model3 = build_model(
    embedding_dim=256,
    lstm_units=128,
    dropout_rate=0.5
)

print("="*60)
print("Model 3")
print("="*60)

model3.summary()

print("\nHyperparameter tuning completed.")