"""
=========================================
TF-IDF (Term Frequency - Inverse Document Frequency)
=========================================

TF  -> How often a word appears
IDF -> How rare the word is

Rare but important words receive
higher scores.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "I love AI",
    "I love Python",
    "AI loves Python",
    "Python is powerful"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

df = pd.DataFrame(
    X.toarray(),
    columns=feature_names
)

print("=" * 70)
print("TF-IDF Matrix")
print("=" * 70)

print(df.round(3))