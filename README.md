# PDF-Powered Assistant

## Overview
The **PDF-Powered Assistant** is a Python-based tool for querying information from PDF documents using state-of-the-art language models and vector search techniques. The system combines document embeddings, retrieval mechanisms, and large language models (LLMs) to deliver precise and context-aware answers to user queries.

---

## Features
- **PDF Processing**: Extracts and processes text from PDF files for structured indexing.
- **Vector Search**: Embeds document content into a vector space for similarity-based retrieval.
- **Custom LLM Support**: Leverages fine-tuned models like Vicuna or Qwen for response generation.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with natural language reasoning.
- **Persistent Vector Storage**: Uses ChromaDB for storing and querying document embeddings.
- **Interactive Querying**: Provides a user-friendly interface for querying indexed documents.

---

## File Descriptions

### 1. `app.py`
The entry point of the application. It provides a user-friendly CLI to query the assistant interactively.

**Key Functionality**:
- Accepts user queries.
- Processes queries using a RetrievalQA chain.
- Outputs the generated response.

---

### 2. `ingest.py`
Handles the ingestion of PDF documents. Processes the document by splitting it into smaller chunks, embedding the text, and storing it in ChromaDB.

**Key Functions**:
- `process_pdf(pdf_path)`: Loads and extracts text from the PDF file.
- `store_documents_in_chroma(documents)`: Splits the document into chunks, converts them into embeddings, and stores them in ChromaDB.

---

### 3. `model-download.py`
Downloads and saves a pre-trained language model locally. This script supports loading models from Hugging Face.

**Key Functionality**:
- Downloads the specified model and tokenizer.
- Saves them to the specified directory.

---

### 4. `query_engine.py`
Defines the logic for query processing. It initializes the retrieval chain and connects the language model with ChromaDB for document-based querying.

**Key Functions**:
- `get_retrieval_chain()`: Creates a RetrievalQA chain using ChromaDB and the pre-trained model.
- `query_chain(query)`: Processes a query through the retrieval chain to generate a response.

---

### 5. `requirements.txt`
Contains all the Python dependencies required for the project:
```plaintext
langchain
chromadb
transformers
sentence-transformers
torch
pymupdf
scikit-learn
python-dotenv
tqdm
llama-cpp-python
langchain-community