"""
=========================================
Stemming in NLP
=========================================

Stemming removes the suffix from a word
to get its root form.

Examples:
Playing  -> play
Played   -> play
Studies  -> studi
Running  -> run
"""

from nltk.stem import PorterStemmer

# Create stemmer object
stemmer = PorterStemmer()

words = [
    "playing",
    "played",
    "plays",
    "studies",
    "running",
    "runner",
    "happiness"
]

print("=" * 50)
print("Stemming Results")
print("=" * 50)

for word in words:
    stem = stemmer.stem(word)
    print(f"{word:<15} ---> {stem}")