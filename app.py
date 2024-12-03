# from query import upsert_sentence, query_sentence

# print("lalalalalala")

# if __name__ == "__main__":
#     # Example: Insert a sentence into Pinecone
#     sentence_to_store = "How can I generate embeddings?"
#     unique_id = "unique-id-1"
#     upsert_sentence(sentence_to_store, unique_id)

#     # Example: Query for similar sentences
#     # query_sentence_text = "How do I create sentence embeddings?"
#     # results = query_sentence(query_sentence_text)
#     # print(f"Query Results: {results}")

#------------------------------------------------
from query import upsert_sentence, query_sentence

# Define a sentence to store
sentence_to_store = "javascript and python are best programming language."
unique_id = "sentence-2"

# Upsert the sentence
# upsert_sentence(sentence_to_store, unique_id)

# Query the sentence
query_sentence("world")



#------------------------------------------------

# from query import upsert_sentence, query_sentence

# # Sentence to insert
# sentence_to_store = "The quick brown fox jumps over the lazy dog."
# unique_id = "sentence-1"
# upsert_sentence(sentence_to_store, unique_id)
# print(f"Inserted sentence: '{sentence_to_store}' with ID: {unique_id}")

# # Sentence to query
# # query = "A fast dark fox leaps over a sleepy dog."
# # results = query_sentence(query)

# # Display query results
# if results and "matches" in results:
#     for match in results["matches"]:
#         print(f"Matched sentence ID: {match['id']} (score: {match['score']:.6f})")
# else:
#     print("No matches found.")