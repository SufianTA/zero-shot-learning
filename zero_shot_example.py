# zero-shot-learning/zero_shot_example.py

from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Example usage
text = "This is a test sentence."
labels = ["positive", "negative", "neutral"]
result = classifier(text, candidate_labels=labels)
print(result)
