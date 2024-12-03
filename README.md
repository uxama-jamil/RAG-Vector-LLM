# Pinecone Sentence Similarity with Embeddings

This project demonstrates how to use Pinecone to store and query sentence embeddings generated from the `sentence-transformers/all-MiniLM-L6-v2` model. The embeddings are created using a process of tokenization, encoding, and mean pooling, and are then normalized and upserted to Pinecone for fast similarity search.

## Project Overview

The application:
- Tokenizes the input sentence.
- Encodes the sentence into embeddings using the pre-trained `sentence-transformers/all-MiniLM-L6-v2` model.
- Applies mean pooling to get a single vector representation.
- Optionally normalizes the embeddings before upserting them into the Pinecone vector database.
- Queries the Pinecone index to retrieve similar sentences based on cosine similarity scores.

## Setup

### Prerequisites
1. Python 3.x
2. Pinecone account and API key: [Sign up here](https://www.pinecone.io/)
3. Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt