# Armwrestling PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions from a PDF document and receive **strict, source-grounded answers** using embeddings, vector search, re-ranking, and an LLM.

---

## Architecture Overview

PDF → Chunking → Embeddings → ChromaDB
                             ↓
User Question → Retrieval → Re-ranking → LLM → Answer + Sources

---

## Project Structure

rag_app/
│
├── app.py
├── rag/
│   ├── loader.py
│   ├── vectorstore.py
│   ├── rag_pipeline.py
│   └── llm.py
├── pdfContent/
│   └── document.pdf
├── templates/
│   └── index.html
├── requirements.txt
└── README.md

---

## Features

- PDF-based question answering
- Semantic search with Sentence Transformers
- Cross-Encoder re-ranking
- ChromaDB vector store
- Strict no-hallucination RAG prompting
- Flask backend + HTML frontend

---

## Tech Stack

- Backend: Flask
- PDF Parsing: PyPDF2
- Embeddings: all-MiniLM-L6-v2
- Re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2
- Vector DB: ChromaDB

---

## Author

Yash Gupta
