"""
====================================================
Hugging Face Pipeline Demo
====================================================
"""

from transformers import pipeline

# ---------------------------------------
# Sentiment Analysis
# ---------------------------------------

sentiment = pipeline(
    "sentiment-analysis"
)

print("="*60)
print("Sentiment Analysis")
print("="*60)

print(
    sentiment(
        "I love learning NLP."
    )
)

# ---------------------------------------
# Summarization
# ---------------------------------------

summarizer = pipeline(
    "summarization"
)

text = """
Artificial Intelligence is transforming healthcare,
education, finance, transportation and robotics.
It enables machines to learn from data,
make predictions,
and automate complex tasks.
"""

summary = summarizer(
    text,
    max_length=40,
    min_length=15,
    do_sample=False
)

print("\nSummary")

print(summary)

# ---------------------------------------
# Translation
# ---------------------------------------

translator = pipeline(
    "translation_en_to_fr"
)

print("\nTranslation")

print(
    translator(
        "Artificial Intelligence is amazing."
    )
)

# ---------------------------------------
# Question Answering
# ---------------------------------------

qa = pipeline(
    "question-answering"
)

context = """
Python is one of the most popular programming
languages used in Data Science and AI.
"""

question = "Which language is used in AI?"

answer = qa(
    question=question,
    context=context
)

print("\nQuestion Answering")

print(answer)