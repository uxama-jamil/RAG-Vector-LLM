import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/Employee-Handbook-Updated.pdf"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")

def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def store_documents_in_chroma(documents):
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    print("\n--- Finished creating vector store ---")
    db.persist()

if __name__ == "__main__":
    docs = process_pdf(PDF_PATH)
    store_documents_in_chroma(docs)
    print(f"Data successfully stored in ChromaDB at {CHROMA_DIR}.")