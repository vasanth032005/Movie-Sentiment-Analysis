"""
=====================================================
IMDB Dataset Loader
=====================================================

Author : Vasanth

Loads the IMDB Movie Review Dataset
included with TensorFlow.

Classes:
0 = Negative
1 = Positive

=====================================================
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Parameters
# -----------------------------

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# -----------------------------
# Load Dataset
# -----------------------------

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("\nExample Label :", y_train[0])

print("\nOriginal Review Length :", len(X_train[0]))

# -----------------------------
# Padding
# -----------------------------

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post"
)

print("\nAfter Padding")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("\nFirst Review (Padded)")

print(X_train[0])