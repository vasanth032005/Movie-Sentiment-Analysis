"""
=========================================
Stop Words Removal
=========================================

Stop words are common words that usually
do not add significant meaning.

Examples:
is
the
a
an
this
that
of
to
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download resources (only first time)
nltk.download("punkt")
nltk.download("stopwords")

text = "I love this movie because it is amazing"

tokens = word_tokenize(text)

stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("=" * 50)
print("Original Sentence")
print("=" * 50)
print(text)

print("\nOriginal Tokens")
print(tokens)

print("\nStop Words Removed")
print(filtered_words)

print("\nFinal Sentence")
print(" ".join(filtered_words))