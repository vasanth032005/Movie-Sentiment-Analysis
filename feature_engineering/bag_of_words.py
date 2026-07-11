"""
=========================================
Bag of Words (BoW)
=========================================

BoW converts text into numerical vectors
based on word frequency.
"""

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love AI",
    "I love Python",
    "AI loves Python",
    "Python is powerful"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print("=" * 60)
print("Vocabulary")
print("=" * 60)
print(vectorizer.get_feature_names_out())

print("\n" + "=" * 60)
print("Bag of Words Matrix")
print("=" * 60)
print(X.toarray())
