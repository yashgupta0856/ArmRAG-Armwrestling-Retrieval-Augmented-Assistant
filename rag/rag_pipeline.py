from rag.vectorstore import collection, embed
from rag.llm import run_llm

def rag(query: str):
    query_emb = embed(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(docs)

    prompt = f"""
Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

If the answer is not present, say it is not mentioned.
"""

    answer = run_llm(prompt)

    sources = [
        {"page": m["page"], "chunk_id": m["chunk_id"]}
        for m in metas
    ]

    return answer, sources
