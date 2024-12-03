import os
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

# Load environment variables
load_dotenv()

# Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "test-db"

print(f"Pinecone API Key: {PINECONE_API_KEY}")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key or Environment not set in .env file.")

# Initialize Pinecone
client = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the index
# if INDEX_NAME not in client.list_indexes():
#     client.create_index(name=INDEX_NAME, dimension=384)

index = client.Index(INDEX_NAME)
# Load the tokenizer and model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)