"""
====================================================
BERT Sentiment Analysis
====================================================
"""

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis"
)

text = "Artificial Intelligence is changing the world."

result = classifier(text)

print("="*60)
print("Sentence")
print("="*60)

print(text)

print("\nPrediction")

print(result)