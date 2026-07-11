"""
=========================================
Lemmatization in NLP
=========================================

Lemmatization converts words into their
dictionary form (lemma).

Examples:
Running -> Run
Cars    -> Car
Studies -> Study
Better  -> Good (with POS tagging)
"""

import nltk
from nltk.stem import WordNetLemmatizer

# Download WordNet (only first time)
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

words = [
    "running",
    "cars",
    "studies",
    "playing"
]

print("=" * 50)
print("Lemmatization Results")
print("=" * 50)

for word in words:
    # pos='v' tells NLTK to treat the word as a verb
    lemma = lemmatizer.lemmatize(word, pos="v")
    print(f"{word:<15} ---> {lemma}")