# MiniRAG PDF Question-Answering System

A lightweight Retrieval-Augmented Generation (RAG) pipeline implemented entirely inside a  **single Jupyter notebook (`notebook.ipynb`)** , allowing users to upload any PDF and ask context-aware questions about its content. This system demonstrates the fundamentals of RAG using:

* **Python** for preprocessing and orchestration
* **PyPDF2** for PDF text extraction
* **SentenceTransformers (MiniLM-L6-v2)** for semantic embeddings
* **ChromaDB** for vector storage and similarity search
* **Ollama (TinyLlama)** as the local LLM for answer generation

This notebook-based workflow provides a clear, linear learning experience with well-structured sections, comments, and outputs.

---

## Project Structure

```
project/
│── notebook.ipynb            # Single consolidated notebook with the full RAG pipeline
│── sample.pdf                # Example PDF (optional)
│── README.md
```

**Note:**

There is **no `models/` folder** because the LLM is executed directly through  **Ollama** , which manages its own model storage.

---

## Pipeline Overview

### **1. PDF Loading**

Extracts raw text from uploaded documents using PyPDF2.

### **2. Chunking**

Splits the extracted text into overlapping segments for better retrieval quality.

### **3. Embedding Generation**

Converts chunks into semantic vectors using `all-MiniLM-L6-v2`.

### **4. Vector Database (ChromaDB)**

Stores the text chunks + embeddings and performs cosine similarity search.

### **5. Query Handling**

User asks a natural-language question inside the notebook.

### **6. Retrieval**

Chroma returns the top-n relevant text chunks.

### **7. Prompt Construction**

Builds a controlled prompt that restricts LLM responses to retrieved context.

### **8. LLM Answering (Ollama)**

TinyLlama generates a grounded answer based purely on the provided context.

---

## Installation

### **1. Install dependencies**

```bash
pip install PyPDF2 chromadb sentence-transformers
```

### **2. Install Ollama**

Download: [https://ollama.com/download](https://ollama.com/download)

Pull the TinyLlama model:

```bash
ollama pull tinyllama
```

---

## Example Output

```
QUESTION: What is the use of the open() function? 

ANSWER:
The open() function is used to open files and returns a file object that allows reading or writing.
```

---

## Author

Created as an educational project to understand the core mechanics of RAG, embeddings, vector search, and local LLM reasoning.
