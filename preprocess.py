# import re

# def preprocess_text(text):
#     """
#     Preprocess the input text to ensure consistent embeddings.
#     - Convert text to lowercase
#     - Remove punctuation
#     - Trim extra spaces
#     """
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text