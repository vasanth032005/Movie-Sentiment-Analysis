"""
=========================================
Word Tokenization
=========================================

Split a sentence into individual words.
"""

import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer (only first time)
nltk.download("punkt")

text = "I love Artificial Intelligence"

tokens = word_tokenize(text)

print("=" * 50)
print("Sentence")
print("=" * 50)
print(text)

print("\n" + "=" * 50)
print("Tokens")
print("=" * 50)

for i, token in enumerate(tokens, start=1):
    print(f"{i}. {token}")

print("\nTotal Tokens :", len(tokens))