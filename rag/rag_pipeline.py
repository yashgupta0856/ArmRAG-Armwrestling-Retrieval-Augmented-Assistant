from rag.vectorstore import model, collection
from rag.llm import run_llm

# due to memory shortage at render platform i have to remove re-ranker 
# from sentence_transformers import CrossEncoder
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rag(query: str):
    # Embed query
    query_emb = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context
    context = "\n\n".join(docs)

    prompt = f"""
Use ONLY the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{query}

RULES:
- If the answer is not in the context, say it is not mentioned.
- Do NOT guess.
"""

    answer = run_llm(prompt)

    sources = [
        {
            "page": m["page"],
            "chunk_id": m["chunk_id"]
        }
        for m in metas
    ]

    return answer, sources
