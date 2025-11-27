# rag_utils.py
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import os

# -------------------------------
# Load PDF → pages
# -------------------------------
def load_file(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


# -------------------------------
# Chunking with metadata
# -------------------------------
def split_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []
    metadatas = []
    global_index = 0

    for p in pages:
        text = p["text"]
        page_no = p["page"]
        text_length = len(text)

        start = 0
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            meta = {
                "chunk_id": f"p{page_no}-c{global_index}",
                "page": page_no,
                "start_char": start,
                "end_char": min(end, text_length)
            }

            chunks.append(chunk)
            metadatas.append(meta)

            global_index += 1
            start = end - overlap

    return chunks, metadatas


# -------------------------------
# Build Vector DB (ChromaDB)
# -------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")   # persistent for deployment
collection = None

def build_vector_db(pdf_path):
    global collection

    pages = load_file(pdf_path)
    chunks, metas = split_into_chunks(pages)

    # Create collection
    collection = chroma_client.get_or_create_collection(
        name="pdf_reader",
        metadata={"hnsw:space": "cosine"}
    )

    # Insert all chunks
    for chunk, meta in zip(chunks, metas):
        emb = model.encode([chunk])[0].tolist()
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[meta["chunk_id"]],
            metadatas=[meta]
        )

    print("Vector DB ready with", len(chunks), "chunks.")


# -------------------------------
# Cross-Encoder for re-ranking
# -------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# RAG Function (used by FastAPI)
# -------------------------------
def rag(query):
    global collection

    # Embed query
    expanded = f"Explain in detail: {query} (in context of armwrestling)"
    query_emb = model.encode([expanded])[0].tolist()

    # Retrieve
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metas  = results["metadatas"][0]

    # Re-rank
    pairs = [(query, c) for c in retrieved_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, retrieved_chunks, retrieved_metas),
        reverse=True,
        key=lambda x: x[0]
    )

    # select top 2 chunks
    final = ranked[:2]
    final_chunks = [x[1] for x in final]
    final_metas  = [x[2] for x in final]

    context = "\n\n".join(final_chunks)

    return context, final_metas
