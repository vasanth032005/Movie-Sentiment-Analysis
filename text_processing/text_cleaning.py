"""
=========================================
Text Cleaning in NLP
=========================================

Tasks:
1. Convert text to lowercase
2. Remove numbers
3. Remove punctuation
4. Remove extra spaces
"""

import re


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


def main():
    text = "I LOVE this Movie!! 😍😍 It is AMAZING 100%"

    print("=" * 50)
    print("Original Text")
    print("=" * 50)
    print(text)

    cleaned = clean_text(text)

    print("\n" + "=" * 50)
    print("Cleaned Text")
    print("=" * 50)
    print(cleaned)


if __name__ == "__main__":
    main()