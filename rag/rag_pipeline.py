from sentence_transformers import CrossEncoder
from rag.vectorstore import model, collection
from rag.llm import run_llm

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rag(query):
    query_emb = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    scores = reranker.predict([(query, d) for d in docs])

    ranked = sorted(zip(scores, docs, metas), reverse=True)
    top_chunks = ranked[:2]

    context = "\n\n".join([c[1] for c in top_chunks])

    prompt = f"""
Use ONLY this context:

{context}

Question:
{query}

If not found, say it is not mentioned.
"""

    answer = run_llm(prompt)

    sources = [
        {
            "page": m["page"],
            "chunk_id": m["chunk_id"],
            "score": float(s)
        }
        for s, _, m in top_chunks
    ]

    return answer, sources
