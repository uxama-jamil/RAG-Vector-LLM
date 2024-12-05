from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

PDF_PATH = "data/Employee-Handbook-Updated.pdf"
def process_pdf(pdf_path):
    """
    Load the PDF using PyMuPDFLoader and return the documents as a list of text.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    document_texts = [doc.page_content for doc in documents]
    print("document:- ",document_texts)
    return documents

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(process_pdf(pdf_path=PDF_PATH))

print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0]}\n")

# process_pdf(pdf_path=PDF_PATH)
# from transformers import AutoTokenizer, AutoModel

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("nvidia/nv-embed-v2")
# model = AutoModel.from_pretrained("nvidia/nv-embed-v2")

# texts = ["This is a sample sentence.", "Embedding text is fun!"]
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1) 

# from sklearn.preprocessing import normalize

# normalized_embeddings = normalize(embeddings.numpy())

# print(normalized_embeddings)
