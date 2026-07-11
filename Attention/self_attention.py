"""
==========================================================
Self Attention From Scratch
==========================================================

Author : Vasanth

Self-Attention allows every word in a sentence
to attend to every other word.

==========================================================
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------------------
# Input Embeddings
# (3 words, embedding dimension = 4)
# ------------------------------------------------

X = np.array([
    [1, 0, 1, 0],   # I
    [0, 2, 0, 2],   # Love
    [1, 1, 1, 1]    # AI
])

print("=" * 60)
print("Input Embeddings")
print("=" * 60)

print(X)

# ------------------------------------------------
# Weight Matrices
# ------------------------------------------------

embedding_dim = 4

WQ = np.random.rand(embedding_dim, embedding_dim)
WK = np.random.rand(embedding_dim, embedding_dim)
WV = np.random.rand(embedding_dim, embedding_dim)

# ------------------------------------------------
# Query, Key, Value
# ------------------------------------------------

Q = X @ WQ
K = X @ WK
V = X @ WV

print("\nQuery Matrix")
print(Q)

print("\nKey Matrix")
print(K)

print("\nValue Matrix")
print(V)

# ------------------------------------------------
# Attention Scores
# ------------------------------------------------

scores = Q @ K.T

print("\nAttention Scores")
print(scores)

# ------------------------------------------------
# Softmax Function
# ------------------------------------------------

def softmax(matrix):
    exp = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

attention_weights = softmax(scores)

print("\nAttention Weights")
print(attention_weights)

# ------------------------------------------------
# Final Output
# ------------------------------------------------

output = attention_weights @ V

print("\nSelf Attention Output")
print(output)