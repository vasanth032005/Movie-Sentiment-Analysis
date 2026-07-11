"""
=========================================
FastText Example
=========================================

FastText learns sub-word information.

It works well for:

Misspelled words

New words

Rare words

Author : Vasanth
"""

from gensim.models import FastText

sentences = [
    ["i", "love", "python"],
    ["python", "is", "easy"],
    ["machine", "learning"],
    ["deep", "learning"],
    ["artificial", "intelligence"]
]

print("=" * 60)
print("Training FastText Model")
print("=" * 60)

model = FastText(
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=1,
    epochs=100
)

print("\nVocabulary")

print(model.wv.index_to_key)

print("\nVector for 'python'")

print(model.wv["python"])

print("\nMost Similar Words")

print(model.wv.most_similar("python"))