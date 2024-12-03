# from config import index
# from embedding import generate_embedding

# def upsert_sentence(sentence, unique_id):
#     """
#     Generate and store an embedding for the given sentence.
#     """
#     embedding = generate_embedding(sentence)
#     print(f"embeddings are {embedding}")
#     vector = [
#         {
#             "id": f"sentence-{hash(sentence)}",  # Create a unique ID based on the sentence
#             "values": embedding[0],  # Embedding vector
#             "metadata": {"sentence": sentence}  # Optional metadata
#         }
#     ]
#     index.upsert(items=[(unique_id, embedding)],vectors=list(vector))
#     print(f"Stored embedding for: {sentence}")

# def query_sentence(sentence, top_k=1):
#     """
#     Query Pinecone for similar embeddings.
#     """
#     # embedding = generate_embedding(sentence)
#     # result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
#     # return result


#------------------------------------------------#------------------------------------------------

from config import index, model, tokenizer
from utils import mean_pooling,normalize_embeddings
import torch

def upsert_sentence(sentence, unique_id):
    """
    Generate an embedding for a sentence and upsert it into the Pinecone index.
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    print('tokenize inputs.....',inputs)

    # Generate embeddings using the model
    with torch.no_grad():
        model_output = model(**inputs)
    print('Generate embeddings using the model......',model_output)
    # Apply mean pooling to get a single vector representation
    embedding = mean_pooling(model_output, inputs["attention_mask"])

    # Normalize the embedding
    embedding = normalize_embeddings(embedding)

    # Convert embedding to a flat list
    embedding = embedding.squeeze(0).numpy().tolist()

    # Define the vector for Pinecone
    vector = [
        {
            "id": unique_id,
            "values": embedding,
            "metadata": {"sentence": sentence}
        }
    ]

    # Upsert into Pinecone
    index.upsert(vectors=vector)
    print(f"Upserted sentence: {sentence}")


def query_sentence(sentence, top_k=3):
    """
    Query the Pinecone index to find similar sentences.
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    # Generate embeddings using the model
    with torch.no_grad():
        model_output = model(**inputs)

    # Apply mean pooling to get a single vector representation
    embedding = mean_pooling(model_output, inputs["attention_mask"])

    # Normalize the embedding
    embedding = normalize_embeddings(embedding)

    # Convert embedding to a flat list
    embedding = embedding.squeeze(0).numpy().tolist()

    # Query the Pinecone index
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True, metric="cosine")

    # Print results
    for match in results["matches"]:
        print(f"Matched sentence: {match['metadata']['sentence']} (score: {match['score']})")


#------------------------------------------------#------------------------------------------------


# from config import index
# from sentence_embeddings import encode_sentence

# def upsert_sentence(sentence, unique_id):
#     """
#     Store a sentence and its embedding in the Pinecone index.
#     """
#     embedding = encode_sentence(sentence)
#     index.upsert([(unique_id, embedding)])  # Insert the sentence and its embedding

# def query_sentence(sentence, top_k=1):
#     """
#     Query the Pinecone index for the most similar sentence.
#     """
#     embedding = encode_sentence(sentence)
#     results = index.query(vector=embedding, top_k=top_k, include_metadata=False)
#     return results