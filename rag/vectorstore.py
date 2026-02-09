import chromadb
import os
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pdf_reader")

def embed(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_vectorstore(chunks, metadatas):
    for chunk, meta in zip(chunks, metadatas):
        emb = embed(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[meta["chunk_id"]],
            metadatas=[meta]
        )
