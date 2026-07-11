"""
==========================================================
Scaled Dot Product Attention
==========================================================
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# -----------------------------
# Query
# -----------------------------

Q = np.array([
    [1,0,1],
    [0,1,0]
])

# -----------------------------
# Key
# -----------------------------

K = np.array([
    [1,1,0],
    [0,1,1]
])

# -----------------------------
# Value
# -----------------------------

V = np.array([
    [10,20],
    [30,40]
])

# -----------------------------
# Dot Product
# -----------------------------

scores = Q @ K.T

print("Scores")

print(scores)

# -----------------------------
# Scale
# -----------------------------

dk = K.shape[1]

scaled_scores = scores / np.sqrt(dk)

print("\nScaled Scores")

print(scaled_scores)

# -----------------------------
# Softmax
# -----------------------------

def softmax(x):

    exp = np.exp(x - np.max(x, axis=1, keepdims=True))

    return exp / np.sum(exp, axis=1, keepdims=True)

weights = softmax(scaled_scores)

print("\nAttention Weights")

print(weights)

# -----------------------------
# Final Output
# -----------------------------

output = weights @ V

print("\nAttention Output")

print(output)