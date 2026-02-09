# Simple in-memory document store

DOCUMENTS = []

def build_store(chunks, metadatas):
    global DOCUMENTS
    DOCUMENTS = list(zip(chunks, metadatas))
