"""
=========================================
Word2Vec Example
=========================================

Word2Vec learns the meaning of words by
looking at the surrounding words.

Author : Vasanth
"""

from gensim.models import Word2Vec

# Training sentences
sentences = [
    ["i", "love", "python"],
    ["python", "is", "easy"],
    ["i", "love", "artificial", "intelligence"],
    ["artificial", "intelligence", "uses", "python"],
    ["deep", "learning", "uses", "python"],
    ["machine", "learning", "is", "powerful"]
]

print("=" * 60)
print("Training Word2Vec Model...")
print("=" * 60)

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=100
)

print("\nVocabulary")
print(model.wv.index_to_key)

print("\nVector for 'python'")
print(model.wv["python"])

print("\nWords Similar to 'python'")
print(model.wv.most_similar("python", topn=3))