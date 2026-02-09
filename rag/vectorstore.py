import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="pdf_reader")

def build_vectorstore(chunks, metadatas):
    for chunk, meta in zip(chunks, metadatas):
        emb = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[meta["chunk_id"]],
            metadatas=[meta]
        )

    return collection
