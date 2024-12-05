import os
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
import torch
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")
MODEL_PATH = os.getenv("MODEL_PATH")

# Load SentenceTransformer for query embedding
# embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Modify the device setup
device = 0 if torch.cuda.is_available() else -1  # GPU if available, else fallback to CPU


def get_retrieval_chain():
    """
    Create and return a RetrievalQA chain that uses a pre-trained model (e.g., Vicuna) and ChromaDB.
    """
    # Load the pre-trained HuggingFace model (e.g., Vicuna or any other LLM)
    # Load Vicuna model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    llm = HuggingFacePipeline(pipeline=pipe)

    # Load ChromaDB
    # db = Chroma(persist_directory=CHROMA_DIR)

     # Load ChromaDB with embedding model
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load ChromaDB with SentenceTransformer embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Create RetrievalQA pipeline
    # retriever = db.as_retriever(search_type="similarity_score_threshold",
    # search_kwargs={"k": 3, "score_threshold": 0.4},)

    # Create the RetrievalQA pipeline
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True # Return both the response and the source documents
    )
    return chain

def query_chain(query):
    """
    Process the input query by retrieving relevant documents and generating a response.
    """
    print(f"Received query: {query}")
    chain = get_retrieval_chain()
    result = chain.invoke(query)
    return result

# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     response = query_chain(query)
#     print(f"Response: {response}")