"""
=========================================
GloVe Embedding Example
=========================================

Author : Vasanth

Download:
glove.6B.100d.txt

Place inside:

models/
    glove.6B.100d.txt
"""

import numpy as np

glove_file = "models/glove.6B.100d.txt"

embeddings = {}

print("Loading GloVe Model...")

with open(glove_file, encoding="utf8") as file:

    for line in file:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:], dtype="float32")

        embeddings[word] = vector

print("Loaded", len(embeddings), "word vectors.")

print("\nVector for 'king'")

print(embeddings["king"])

print("\nVector Length")

print(len(embeddings["king"]))